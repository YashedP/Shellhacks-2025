import ee
import requests
import zipfile
import io
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Tuple
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
EXPORT_SCALE = 30  # default Landsat scale
TILE_SCALE_METERS = 50000
DEBUG_ONE_TILE = True
MAX_WORKERS = 4

# Retry configuration
MAX_RETRIES = 7   # cap retries to avoid infinite loops
BASE_DELAY = 1
MAX_DELAY = 300
BACKOFF_MULTIPLIER = 2

FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_tiles.txt")
RERUN_SCRIPT = os.path.join(OUTPUT_DIR, "rerun_failed_tiles.py")

# ================================
# RETRY DECORATOR
# ================================
def retry_with_exponential_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY,
                                   max_delay=MAX_DELAY, backoff_multiplier=BACKOFF_MULTIPLIER):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt, delay = 0, base_delay
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    # Bail early if this is a “too big” error
                    if "must be less than or equal to" in str(e):
                        raise e
                    jitter = random.uniform(0, 1)
                    actual_delay = min(delay + jitter, max_delay)
                    thread_safe_print(f"[{func.__name__}] Attempt {attempt} failed: {e}")
                    thread_safe_print(f"Retrying in {actual_delay:.2f} seconds...")
                    time.sleep(actual_delay)
                    delay = min(delay * backoff_multiplier, max_delay)
            raise Exception(f"{func.__name__} failed after {attempt} retries")
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
# NDWI HELPERS
# ================================
def add_ndwi_landsat(image):
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands(ndwi)

def add_ndwi_sentinel(image):
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")  # 10m bands
    return image.addBands(ndwi)

# ================================
# GRID CREATION
# ================================
def make_grid(region, scale=TILE_SCALE_METERS):
    lonlat = ee.Image.pixelLonLat()
    lon = lonlat.select("longitude").divide(scale).floor()
    lat = lonlat.select("latitude").divide(scale).floor()
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
# HELPERS
# ================================
def get_tile_filename(geom, year):
    centroid = geom.centroid().coordinates().getInfo()
    lon = round(centroid[0], 5)
    lat = round(centroid[1], 5)
    return os.path.join(OUTPUT_DIR, f"{lat}_{lon}_{year}.tif")

def log_failed_tile(geom, year, error_msg):
    centroid = geom.centroid().coordinates().getInfo()
    lon = round(centroid[0], 5)
    lat = round(centroid[1], 5)
    
    # Flush error to console first
    thread_safe_print(f"FAILED TILE: {lat}_{lon}_{year} - Error: {error_msg}")
    
    # Write to failed tiles log for manual rerun
    with open(FAILED_LOG, "a") as f:
        f.write(f"{year},{lat},{lon},{error_msg}\n")
    
    thread_safe_print(f"Logged failed tile to {FAILED_LOG} for manual rerun")

# ================================
# DOWNLOAD FUNCTION
# ================================
@retry_with_exponential_backoff()
def download_tile_with_retry(img, geom, out_file, scale=EXPORT_SCALE):
    url = img.getDownloadURL({
        "scale": scale,
        "crs": "EPSG:4326",
        "region": geom.getInfo()   # use polygon, not bounds
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
    try:
        return download_tile_with_retry(img, geom, out_file, scale)
    except Exception as e:
        thread_safe_print(f"Final error for {out_file}: {e}")
        log_failed_tile(geom, os.path.basename(out_file).split("_")[-1].split(".")[0], str(e))
        return False

# ================================
# PROCESS SINGLE YEAR
# ================================
def process_year(year: int, geom) -> Tuple[int, bool]:
    try:
        thread_safe_print(f"Processing {year}...")

        if year < 2015:
            dataset = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                        .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
                        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
                        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                        .filterBounds(geom)
                        .filterDate(f"{year}-01-01", f"{year}-12-31")
                        .map(add_ndwi_landsat))
            scale = 30
        else:
            dataset = (ee.ImageCollection("COPERNICUS/S2_SR")
                        .filterBounds(geom)
                        .filterDate(f"{year}-01-01", f"{year}-12-31")
                        .map(add_ndwi_sentinel))
            scale = 10

        if dataset.size().getInfo() == 0:
            thread_safe_print(f"No data for {year}, skipping...")
            return (year, False)

        ndwi_median = dataset.select("NDWI").median().clip(geom)
        out_file = get_tile_filename(geom, year)

        if os.path.exists(out_file):
            thread_safe_print(f"Already exists: {out_file}")
            return (year, True)

        success = download_tile(ndwi_median, geom, out_file, scale=scale)
        return (year, success)
    except Exception as e:
        thread_safe_print(f"Error processing {year}: {e}")
        log_failed_tile(geom, year, str(e))
        return (year, False)

# ================================
# PARALLEL PROCESSING
# ================================
def process_years_parallel(geom, start_year: int, end_year: int):
    years_to_process = []
    for year in range(start_year, end_year + 1):
        out_file = get_tile_filename(geom, year)
        if not os.path.exists(out_file):
            years_to_process.append(year)

    if not years_to_process:
        thread_safe_print("All files already exist!")
        return

    thread_safe_print(f"Processing {len(years_to_process)} years with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_year = {executor.submit(process_year, year, geom): year for year in years_to_process}
        completed, failed = 0, 0
        for future in as_completed(future_to_year):
            year, success = future.result()
            completed += 1
            if not success:
                failed += 1
            thread_safe_print(f"Progress: {completed}/{len(years_to_process)} years done")

    thread_safe_print(f"Complete! {completed - failed} successful, {failed} failed")

# ================================
# MAIN LOOP
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DEBUG_ONE_TILE:
    tile = ee.Feature(tile_list.get(0))
    geom = tile.geometry()
    start_year = 1984
    current_year = datetime.datetime.now().year
    process_years_parallel(geom, start_year, current_year)

print("Done!")
