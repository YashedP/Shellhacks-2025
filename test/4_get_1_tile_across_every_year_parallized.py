import ee
import requests
import zipfile
import io
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple

# ================================
# CONFIG
# ================================
SERVICE_ACCOUNT = "shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com"
KEY_FILE = "gcp_key.json"
PROJECT = "striped-orbit-473405-h0"
OUTPUT_DIR = "florida_tiles"
EXPORT_SCALE = 30
DEBUG_ONE_TILE = True
MAX_WORKERS = 4  # Number of parallel download threads

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
def make_grid(region, scale=50000):
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

tiles = make_grid(florida.geometry(), 50000)
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
def download_tile(img, geom, out_file, scale=EXPORT_SCALE):
    try:
        url = img.getDownloadURL({
            "scale": scale,
            "crs": "EPSG:4326",
            "region": geom.bounds().getInfo()  # use bounding box for stability
        })
        thread_safe_print(f"Downloading {out_file} ...")
        response = requests.get(url)
        if response.status_code != 200:
            thread_safe_print(f"Failed for {out_file}: {response.text[:200]}")
            return False
        z = zipfile.ZipFile(io.BytesIO(response.content))
        for fname in z.namelist():
            if fname.endswith(".tif"):
                z.extract(fname, OUTPUT_DIR)
                os.rename(os.path.join(OUTPUT_DIR, fname), out_file)
        thread_safe_print(f"Saved {out_file}")
        return True
    except Exception as e:
        thread_safe_print(f"Error for {out_file}: {e}")
        return False

# ================================
# PROCESS SINGLE YEAR
# ================================
def process_year(year: int, all_landsat, geom) -> Tuple[int, bool]:
    """Process a single year and return (year, success)"""
    try:
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
    except Exception as e:
        thread_safe_print(f"Error processing {year}: {e}")
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
