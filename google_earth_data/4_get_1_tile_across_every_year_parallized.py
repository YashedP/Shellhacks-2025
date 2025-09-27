import ee
import requests
import zipfile
import io
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple
import time
import random
from functools import wraps

# ================================
# CONFIG
# ================================
SERVICE_ACCOUNT = "shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com"
KEY_FILE = "gcp_key.json"
PROJECT = "striped-orbit-473405-h0"
OUTPUT_DIR = "florida_tiles"
EXPORT_SCALE = 30
TILE_SCALE_METERS = 50000
DEBUG_ONE_TILE = True
MAX_WORKERS = 4  # Number of parallel download threads

# Retry configuration
MAX_RETRIES = float('inf')  # Never stop retrying
BASE_DELAY = 1  # Base delay in seconds
MAX_DELAY = 300  # Maximum delay in seconds (5 minutes)
BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# ================================
# EXPONENTIAL BACKOFF RETRY DECORATOR
# ================================
def retry_with_exponential_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, 
                                 max_delay=MAX_DELAY, backoff_multiplier=BACKOFF_MULTIPLIER):
    """
    Decorator that implements exponential backoff retry logic.
    Will retry indefinitely until success.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    # Calculate delay with jitter to avoid thundering herd
                    jitter = random.uniform(0, 1)
                    actual_delay = min(delay + jitter, max_delay)
                    
                    thread_safe_print(f"Attempt {attempt} failed for {func.__name__}: {e}")
                    thread_safe_print(f"Retrying in {actual_delay:.2f} seconds...")
                    
                    time.sleep(actual_delay)
                    
                    # Exponential backoff: increase delay for next attempt
                    delay = min(delay * backoff_multiplier, max_delay)
                    
            return None  # This should never be reached
        return wrapper
    return decorator

# ================================
# AUTHENTICATE
# ================================
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials, project=PROJECT)

# ================================
# LOAD FLORIDA BOUNDARY
# ================================
florida = ee.FeatureCollection("TIGER/2018/States") \
             .filter(ee.Filter.eq("STUSPS", "FL"))

# ================================
# NDWI HELPER
# ================================
def add_ndwi(image):
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands(ndwi)

# ================================
# GRID CREATION
# ================================
def make_grid(region, scale=TILE_SCALE_METERS):
    lonlat = ee.Image.pixelLonLat()
    lon = lonlat.select("longitude").divide(scale).floor()
    lat = lonlat.select("latitude").divide(scale).floor()
    # Combine lon/lat into one band
    grid_id = lon.multiply(100000).add(lat).toInt()
    grid = grid_id.reduceToVectors(
        geometry=region,
        scale=scale,
        geometryType="polygon",
        eightConnected=False,
        labelProperty="cell_id"
    )
    return grid

tiles = make_grid(florida.geometry(), TILE_SCALE_METERS)
tile_list = tiles.toList(tiles.size())

print("Number of tiles:", tiles.size().getInfo())

# ================================
# THREAD-SAFE PRINTING
# ================================
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# ================================
# DOWNLOAD FUNCTION
# ================================
@retry_with_exponential_backoff()
def download_tile_with_retry(img, geom, out_file, scale=EXPORT_SCALE):
    """Download tile with automatic retry on failure"""
    url = img.getDownloadURL({
        "scale": scale,
        "crs": "EPSG:4326",
        "region": geom.bounds().getInfo()  # use bounding box for stability
    })
    thread_safe_print(f"Downloading {out_file} ...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
    
    z = zipfile.ZipFile(io.BytesIO(response.content))
    for fname in z.namelist():
        if fname.endswith(".tif"):
            z.extract(fname, OUTPUT_DIR)
            os.rename(os.path.join(OUTPUT_DIR, fname), out_file)
    thread_safe_print(f"Saved {out_file}")
    return True

def download_tile(img, geom, out_file, scale=EXPORT_SCALE):
    """Wrapper function that calls the retry-enabled download function"""
    try:
        return download_tile_with_retry(img, geom, out_file, scale)
    except Exception as e:
        thread_safe_print(f"Final error for {out_file}: {e}")
        return False

# ================================
# PROCESS SINGLE YEAR
# ================================
@retry_with_exponential_backoff()
def process_year_with_retry(year: int, all_landsat, geom) -> Tuple[int, bool]:
    """Process a single year with automatic retry on failure"""
    thread_safe_print(f"Processing {year}...")
    dataset = all_landsat.filterBounds(geom) \
                         .filterDate(f"{year}-01-01", f"{year}-12-31") \
                         .map(add_ndwi)

    if dataset.size().getInfo() == 0:
        thread_safe_print(f"No data for {year}, skipping...")
        return (year, False)

    ndwi_median = dataset.select("NDWI").median().clip(geom)

    out_file = os.path.join(OUTPUT_DIR, f"tile_{year}.tif")
    if os.path.exists(out_file):
        thread_safe_print(f"Already exists: {out_file}")
        return (year, True)

    success = download_tile(ndwi_median, geom, out_file)
    return (year, success)

def process_year(year: int, all_landsat, geom) -> Tuple[int, bool]:
    """Process a single year and return (year, success)"""
    try:
        return process_year_with_retry(year, all_landsat, geom)
    except Exception as e:
        thread_safe_print(f"Final error processing {year}: {e}")
        return (year, False)

# ================================
# PARALLEL PROCESSING
# ================================
def process_years_parallel(all_landsat, geom, start_year: int, end_year: int):
    """Process multiple years in parallel"""
    # Create list of years to process
    years_to_process = []
    for year in range(start_year, end_year + 1):
        out_file = os.path.join(OUTPUT_DIR, f"tile_{year}.tif")
        if not os.path.exists(out_file):
            years_to_process.append(year)
    
    if not years_to_process:
        thread_safe_print("All files already exist!")
        return
    
    thread_safe_print(f"Processing {len(years_to_process)} years in parallel with {MAX_WORKERS} workers...")
    
    # Process years in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_year = {
            executor.submit(process_year, year, all_landsat, geom): year 
            for year in years_to_process
        }
        
        # Process completed tasks
        completed = 0
        failed = 0
        for future in as_completed(future_to_year):
            year, success = future.result()
            completed += 1
            if not success:
                failed += 1
            thread_safe_print(f"Progress: {completed}/{len(years_to_process)} years completed")

    thread_safe_print(f"Parallel processing complete! {completed - failed} successful, {failed} failed")

# ================================
# MAIN LOOP
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DEBUG_ONE_TILE:
    tile = ee.Feature(tile_list.get(0))
    geom = tile.geometry()

    # Landsat collections (Surface Reflectance, harmonized bands)
    landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    landsat7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

    # Merge all into one collection
    all_landsat = landsat5.merge(landsat7).merge(landsat8).merge(landsat9)

    # Process years in parallel
    start_year = 1984
    current_year = datetime.datetime.now().year
    
    process_years_parallel(all_landsat, geom, start_year, current_year)

print("Done!")
