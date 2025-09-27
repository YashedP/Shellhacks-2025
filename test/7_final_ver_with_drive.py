import ee
import os
import datetime
import threading

# ================================
# CONFIG
# ================================
SERVICE_ACCOUNT = "shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com"
KEY_FILE = "gcp_key.json"
PROJECT = "striped-orbit-473405-h0"

OUTPUT_DIR = "florida_tiles"
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_tiles.txt")

EXPORT_SCALE_LANDSAT = 30
EXPORT_SCALE_SENTINEL = 10
TILE_SCALE_METERS = 50000
DEBUG_ONE_TILE = False

# Which Drive folder? (change if you want to export to a subfolder)
DRIVE_FOLDER = "EarthEngineTiles"

# ================================
# AUTHENTICATE
# ================================
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Authenticate()
ee.Initialize()
# ee.Initialize(credentials, project=PROJECT)

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
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
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
    centroid = geom.centroid(maxError=1).coordinates().getInfo()
    lon = round(centroid[0], 5)
    lat = round(centroid[1], 5)
    return f"{lat}_{lon}_{year}"

def log_failed_tile(geom, year, error_msg):
    try:
        centroid = geom.centroid(maxError=1).coordinates().getInfo()
        lon = round(centroid[0], 5)
        lat = round(centroid[1], 5)
        tile_id = f"{lat}_{lon}_{year}"
    except Exception:
        tile_id = f"unknown_{year}"

    thread_safe_print(f"FAILED TILE: {tile_id} - Error: {error_msg}")

    with open(FAILED_LOG, "a") as f:
        f.write(f"{tile_id},{error_msg}\n")

# ================================
# EXPORT TO DRIVE
# ================================
def export_to_drive(img, geom, year, scale, description):
    try:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=description,
            folder=DRIVE_FOLDER,
            fileNamePrefix=description,
            scale=scale,
            region=geom,
            crs="EPSG:4326",
            maxPixels=1e13  # very large limit to allow big images
        )
        task.start()
        thread_safe_print(f"Started export task for {description} → Drive/{DRIVE_FOLDER}")
        return True
    except Exception as e:
        log_failed_tile(geom, year, str(e))
        return False

# ================================
# PROCESS SINGLE YEAR
# ================================
def process_year(year: int, geom):
    try:
        thread_safe_print(f"Processing {year}...")

        if year < 2015:
            dataset = (all_landsat
                        .filterBounds(geom)
                        .filterDate(f"{year}-01-01", f"{year}-12-31")
                        .map(add_ndwi_landsat))
            scale = EXPORT_SCALE_LANDSAT
        else:
            dataset = (all_sentinel
                        .filterBounds(geom)
                        .filterDate(f"{year}-01-01", f"{year}-12-31")
                        .map(add_ndwi_sentinel))
            scale = EXPORT_SCALE_SENTINEL

        if dataset.size().getInfo() == 0:
            thread_safe_print(f"No data for {year}, skipping...")
            return False

        ndwi_median = dataset.select("NDWI").median().clip(geom)
        description = get_tile_filename(geom, year)

        export_to_drive(ndwi_median, geom, year, scale, description)
        return True
    except Exception as e:
        log_failed_tile(geom, year, str(e))
        return False

# ================================
# COLLECTIONS
# ================================
landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
landsat7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
all_landsat = landsat5.merge(landsat7).merge(landsat8).merge(landsat9)

all_sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# ================================
# MAIN LOOP
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DEBUG_ONE_TILE:
    tile = ee.Feature(tile_list.get(0))
    geom = tile.geometry()
    start_year = 1984
    current_year = datetime.datetime.now().year
    for year in range(start_year, current_year + 1):
        process_year(year, geom)

print("All export tasks submitted. Go to https://code.earthengine.google.com/tasks to monitor.")
