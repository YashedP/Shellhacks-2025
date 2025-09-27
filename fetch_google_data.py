import ee
import geemap

# Initialize Earth Engine
ee.Initialize()

# --- Step 1: Define Florida boundary ---
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq("NAME", "Florida"))

# Create a ~5 km coastal buffer (adjust as needed)
coast_buffer = florida.geometry().buffer(5000)

# --- Step 2: Load Landsat Collections ---
def mask_landsat_sr(image):
    # Mask out clouds using pixel quality band (QA_PIXEL)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0) \
        .And(qa.bitwiseAnd(1 << 4).eq(0)) # Clear + water
    return image.updateMask(mask)

# Merge Landsat 5, 7, 8 collections
landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2") \
    .filterBounds(florida) \
    .filterDate("1984-01-01", "2012-05-05") \
    .map(mask_landsat_sr)

landsat7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
    .filterBounds(florida) \
    .filterDate("1999-01-01", "2022-12-31") \
    .map(mask_landsat_sr)

landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(florida) \
    .filterDate("2013-01-01", "2025-12-31") \
    .map(mask_landsat_sr)

landsat = landsat5.merge(landsat7).merge(landsat8)

# --- Step 3: Compute NDWI (Normalized Difference Water Index) ---
def add_ndwi(image):
    # Green band: B2 (Landsat 8), B1 (Landsat 5/7)
    # NIR band:   B5 (Landsat 8), B4 (Landsat 5/7)
    green = ee.Algorithms.If(image.bandNames().contains('SR_B2'), 
                             image.select('SR_B2'), image.select('SR_B1'))
    nir = ee.Algorithms.If(image.bandNames().contains('SR_B5'), 
                           image.select('SR_B5'), image.select('SR_B4'))
    ndwi = ee.Image.normalizedDifference([green, nir]).rename("NDWI")
    return image.addBands(ndwi)

landsat = landsat.map(add_ndwi)

# --- Step 4: Annual composites ---
def annual_composite(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    yearly = landsat.filterDate(start, end).median()
    return yearly.set("year", year)

years = list(range(1984, 2025))
annual_images = ee.ImageCollection.fromImages([annual_composite(y) for y in years])

# --- Step 5: Extract shoreline (NDWI threshold ~0) ---
def extract_shoreline(img):
    water = img.select("NDWI").gt(0)  # water mask
    shoreline = water.reduceToVectors(
        geometry=coast_buffer,
        scale=30, # Landsat resolution
        geometryType='polygon',
        eightConnected=False,
        labelProperty='water',
        reducer=ee.Reducer.countEvery()
    )
    return shoreline.set("year", img.get("year"))

shorelines = annual_images.map(extract_shoreline)

# --- Step 6: Export Example Year (e.g., 2020 shoreline) ---
example = annual_images.filter(ee.Filter.eq("year", 2020)).first()
ndwi_map = example.select("NDWI")

Map = geemap.Map(center=[27.7, -81.5], zoom=6)  # Center on Florida
Map.addLayer(ndwi_map, {"min": -1, "max": 1, "palette": ["brown", "blue"]}, "NDWI 2020")
Map.addLayer(florida, {}, "Florida Boundary")
Map
