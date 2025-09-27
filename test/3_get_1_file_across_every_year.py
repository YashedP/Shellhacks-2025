import ee
import requests
import zipfile
import io
import os
import datetime

# ================================
# CONFIG
# ================================
SERVICE_ACCOUNT = "shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com"
KEY_FILE = "gcp_key.json"
PROJECT = "striped-orbit-473405-h0"
OUTPUT_DIR = "florida_tiles"
EXPORT_SCALE = 30
DEBUG_ONE_TILE = True   # we only do tile[0] but for all years

# ================================
# AUTH
# ================================
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials, project=PROJECT)

# ================================
# REGION
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
# DOWNLOAD FUNCTION
# ================================
def download_tile(img, geom, out_file, scale=EXPORT_SCALE):
    try:
        url = img.getDownloadURL({
            "scale": scale,
            "crs": "EPSG:4326",
            "region": geom.bounds().getInfo()  # use bounding box for stability
        })
        print(f"Downloading {out_file} ...")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed for {out_file}: {response.text[:200]}")
            return
        z = zipfile.ZipFile(io.BytesIO(response.content))
        for fname in z.namelist():
            if fname.endswith(".tif"):
                z.extract(fname, OUTPUT_DIR)
                os.rename(os.path.join(OUTPUT_DIR, fname), out_file)
        print(f"Saved {out_file}")
    except Exception as e:
        print(f"Error for {out_file}: {e}")

# ================================
# MAIN LOOP (one tile, all years)
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

    # Loop years 1984 → now
    start_year = 1984
    current_year = datetime.datetime.now().year

    for year in range(start_year, current_year + 1):
        print(f"Processing {year}...")
        dataset = all_landsat.filterBounds(geom) \
                             .filterDate(f"{year}-01-01", f"{year}-12-31") \
                             .map(add_ndwi)

        if dataset.size().getInfo() == 0:
            print(f"No data for {year}, skipping...")
            continue

        ndwi_median = dataset.select("NDWI").median().clip(geom)

        out_file = os.path.join(OUTPUT_DIR, f"tile_{year}.tif")
        if os.path.exists(out_file):
            print(f"Already exists: {out_file}")
            continue

        download_tile(ndwi_median, geom, out_file)

print("Done!")
