import sys
import ee
import os
import datetime
import threading

# ================================
# CONFIG
# ================================
PROJECT = "striped-orbit-473405-h0"
DRIVE_FOLDER = "EarthEngineTiles"
OUTPUT_DIR = "florida_tiles"
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_tiles.txt")

# Grid & coastline
TILE_SIZE_M = 10000          # 10 km squares (shrink/increase to change file size / count)
COAST_BUFFER_M = 10000       # 10 km band hugging Florida’s boundary

# Resolution policy
UNIFY_TO_30M = True          # True => export all years at 30 m (smaller files, consistent)
SENTINEL_SCALE = 10          # used only if UNIFY_TO_30M=False
LANDSAT_SCALE = 30

DEBUG_ONE_TILE = False

# ================================
# AUTH
# ================================
import ee
ee.Authenticate()
ee.Initialize(project=PROJECT)

# ================================
# THREAD-SAFE PRINT
# ================================
print_lock = threading.Lock()
def ts_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# ================================
# HELPERS
# ================================
def add_ndwi_landsat(image):
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands(ndwi)

def add_ndwi_sentinel(image):
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands(ndwi)

def get_tile_filename(geom, year):
    c = geom.centroid(maxError=1).coordinates().getInfo()
    lon = round(c[0], 5); lat = round(c[1], 5)
    return f"{lat}_{lon}_{year}"

def log_failed_tile(geom, year, msg):
    try:
        c = geom.centroid(maxError=1).coordinates().getInfo()
        lon = round(c[0], 5); lat = round(c[1], 5)
        tid = f"{lat}_{lon}_{year}"
    except Exception:
        tid = f"unknown_{year}"
    ts_print(f"FAILED TILE: {tid} - {msg}")
    with open(FAILED_LOG, "a") as f:
        f.write(f"{tid},{msg}\n")

# ================================
# DATASETS (create once)
# ================================
landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
landsat7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
ALL_LANDSAT = landsat5.merge(landsat7).merge(landsat8).merge(landsat9)

S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# ================================
# REGION: FL + COASTAL STRIP ONLY
# ================================
florida_fc = ee.FeatureCollection("TIGER/2018/States") \
    .filter(ee.Filter.eq("STUSPS", "FL"))

florida = florida_fc.geometry()  # use when you just need the full geometry
# Create coastal strip ~10 km wide
outer = florida.buffer(COAST_BUFFER_M)       # expand outward
inner = florida.buffer(-COAST_BUFFER_M/2)    # shrink inward
boundary_ring = outer.difference(inner)      # narrow strip hugging coast

# Baseline water (use recent L8 median) → intersects boundary buffer
baseline = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(florida)
            .filterDate("2019-01-01","2021-12-31")
            .map(add_ndwi_landsat)
            .select("NDWI").median())

water = baseline.gt(0).selfMask()
boundary_ring = florida.boundary().buffer(COAST_BUFFER_M)      # narrow ring hugging boundary
coastal_region = water.clip(boundary_ring).geometry()          # water near boundary = coastline band

# ================================
# METRIC GRID (EPSG:3857) over coastal region
# ================================
def make_metric_grid(region, tile_m=TILE_SIZE_M):
    proj = ee.Projection("EPSG:3857")
    coords = ee.Image.pixelCoordinates(proj).reproject(proj.atScale(tile_m))
    x = coords.select("x").divide(tile_m).floor()
    y = coords.select("y").divide(tile_m).floor()
    grid_id = x.multiply(1e7).add(y).toInt()
    grid = grid_id.reduceToVectors(
        geometry=region,
        scale=tile_m,
        geometryType="polygon",
        eightConnected=False,
        labelProperty="cell_id",
        crs="EPSG:3857",
        maxPixels=1e13
    )
    return grid

tiles = make_metric_grid(coastal_region, TILE_SIZE_M)
tile_list = tiles.toList(tiles.size())
num_tiles = tiles.size().getInfo()
ts_print(f"Coastal tiles: {num_tiles}")
sys.exit()
# ================================
# EXPORT
# ================================
def export_to_drive(img, geom, year, scale, description):
    try:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=description,
            folder=DRIVE_FOLDER,
            fileNamePrefix=description,
            region=geom,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True}  # compressed, tiled (smaller/faster)
        )
        task.start()
        ts_print(f"Started: {description} → Drive/{DRIVE_FOLDER}")
        return True
    except Exception as e:
        log_failed_tile(geom, year, str(e))
        return False

def process_year(year, geom):
    try:
        if year < 2015:
            coll = (ALL_LANDSAT.filterBounds(geom)
                    .filterDate(f"{year}-01-01", f"{year}-12-31")
                    .map(add_ndwi_landsat))
            scale = LANDSAT_SCALE
        else:
            coll = (S2.filterBounds(geom)
                    .filterDate(f"{year}-01-01", f"{year}-12-31")
                    .map(add_ndwi_sentinel))
            scale = LANDSAT_SCALE if UNIFY_TO_30M else SENTINEL_SCALE

        if coll.size().getInfo() == 0:
            ts_print(f"No data {year}, skip")
            return False

        ndwi = coll.select("NDWI").median().clip(geom)

        # Optional: scale to Int16 for smaller files (lossless)
        # values are typically -1..1; scale by 10000
        ndwi_int16 = ndwi.multiply(10000).toInt16().rename("NDWI")

        desc = get_tile_filename(geom, year)
        return export_to_drive(ndwi_int16, geom, year, scale, desc)
    except Exception as e:
        log_failed_tile(geom, year, str(e))
        return False

# ================================
# MAIN
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
start_year = 1984
end_year = datetime.datetime.now().year

if DEBUG_ONE_TILE:
    tile = ee.Feature(tile_list.get(0))
    geom = tile.geometry()
    for y in range(start_year, end_year + 1):
        process_year(y, geom)
else:
    for i in range(num_tiles):
        tile = ee.Feature(tile_list.get(i))
        geom = tile.geometry()
        for y in range(start_year, end_year + 1):
            process_year(y, geom)

ts_print("All export tasks submitted. Monitor at https://code.earthengine.google.com/tasks")
