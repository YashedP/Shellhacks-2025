import rasterio
import matplotlib.pyplot as plt

tif_path = "florida_tiles/tile_test/download.NDWI.tif"

with rasterio.open(tif_path) as src:
    ndwi = src.read(1)

# Apply water mask (NDWI > 0)
water_mask = ndwi > 0

plt.imshow(water_mask, cmap="Blues")
plt.title("Water Mask (NDWI > 0)")
plt.show()
