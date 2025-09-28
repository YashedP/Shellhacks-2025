import geopandas as gpd
import os, json, numpy as np
import matplotlib.pyplot as plt
# Visualize the predicted coastline rollouts from the GeoPackage file


segment = 1
visfile = rf"output_segments\segment_{segment}_rollout.gpkg"
gdf = gpd.read_file(visfile)


print(gdf.head())
print(gdf.crs)
gdf.plot( legend=True, cmap="viridis", figsize=(10, 8))

plt.title(f"Predicted Coastline Rollout (Segment {segment})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()