import ee
import requests
import zipfile
import io
import os

# ================================
# CONFIGURATION
# ================================
SERVICE_ACCOUNT = "shellhacks@striped-orbit-473405-h0.iam.gserviceaccount.com"
KEY_FILE = "gcp_key.json"   # path to your downloaded JSON key
PROJECT = "striped-orbit-473405-h0"
OUTPUT_DIR = "florida_tiles"
TILE_SCALE_METERS = 50000   # tile size in meters (~50 km)
EXPORT_SCALE = 30           # output resolution in meters

DEBUG_ONE_TILE = True       # ⚡ flip this to False for full run

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
# LOAD LANDSAT 8 COLLECTION 2 & NDWI
# ================================
dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
            .filterBounds(florida) \
            .filterDate("2020-01-01", "2020-12-31")

def add_ndwi(image):
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands(ndwi)

landsat = dataset.map(add_ndwi)
ndwi_median = landsat.select("NDWI").median().clip(florida)

# ================================
# CREATE TILES (GRID OVER FLORIDA)
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
# DOWNLOAD FUNCTION
# ================================
def download_tile(img, geom, out_path, scale=EXPORT_SCALE):
    url = img.getDownloadURL({
        "scale": scale,
        "crs": "EPSG:4326",
        "region": geom.getInfo()
    })
    print(f"Downloading {out_path} ...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed: {response.text[:200]}")
        return
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(out_path)
    print(f"Saved to {out_path}")

# ================================
# MAIN LOOP
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DEBUG_ONE_TILE:
    # Just one tile for local testing
    tile = ee.Feature(tile_list.get(0))
    geom = tile.geometry()
    out_folder = os.path.join(OUTPUT_DIR, "tile_test")
    os.makedirs(out_folder, exist_ok=True)
    download_tile(ndwi_median, geom, out_folder)
else:
    # Full batch of tiles
    num_tiles = tiles.size().getInfo()
    for i in range(num_tiles):
        tile = ee.Feature(tile_list.get(i))
        geom = tile.geometry()
        out_folder = os.path.join(OUTPUT_DIR, f"tile_{i}")
        os.makedirs(out_folder, exist_ok=True)
        download_tile(ndwi_median, geom, out_folder)

print("✅ Done!")
