import rasterio
import matplotlib.pyplot as plt

tif_path = "florida_tiles/tile_1984.tif"

with rasterio.open(tif_path) as src:
    ndwi = src.read(1)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    
    print(f"Insert these two numbers into google maps to get the location of the tile: {src.bounds.left}, {src.bounds}")

# Apply water mask (NDWI > 0)
water_mask = ndwi > 0

plt.imshow(water_mask, cmap="Blues")
plt.title("Water Mask (NDWI > 0)")
plt.show()
