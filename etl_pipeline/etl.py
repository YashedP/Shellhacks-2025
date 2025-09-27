"""
Downloads data from the "Satellite-derived shorelines for the U.S. Atlantic
coast (1984-2021) dataset" and processes it for use in training a predictive model.
Written by Joshua Sheldon
On September 27, 2025
"""

# ----- IMPORTS -----

import os
import shutil
import zipfile

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union
from tqdm import tqdm

# ----- CONFIG -----

kmz_url = "https://cmgds.marine.usgs.gov/data-releases/media/2022/10.5066-P9BQQTCI/555bff79a2e14785a1347c7ad285c8a0/SatelliteDerivedShorelines_FL.kmz"
kmz_name = "SatelliteDerivedShorelines_FL.kmz"
kml_name = "CoastSat_shorelines_FL.kml"
shp_dir = "CoastSat_shorelines_FL"
segment_height = 0.01
output_dir = "segment_gdfs"

# ----- LOGIC -----

# First, download the data
def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


# If file isn't already there, download it
try:
    open(kmz_name, "rb").close()
    print(f"{kmz_name} already exists, skipping download.")
except FileNotFoundError:
    print(f"Downloading {kmz_name}...")
    download_file(kmz_url, kmz_name)

# If file hasn't already been unzipped, unzip it
# (We can tell if file has been unzipped if kml_name exists)
try:
    open(kml_name, "rb").close()
    print(f"{kml_name} already exists, skipping unzip.")
except FileNotFoundError:
    print(f"Unzipping {kmz_name}...")
    with zipfile.ZipFile(kmz_name, "r") as zip_ref:
        zip_ref.extractall(".")

# Eliminate single point geometries from the KML and
# save the result as a shapefile if shp_dir DNE

# enable KML support if not enabled
fiona.drvsupport.supported_drivers["kml"] = "rw"  # type: ignore
fiona.drvsupport.supported_drivers["KML"] = "rw"  # type: ignore
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"  # type: ignore

# Test if dir exists
if os.path.isdir(shp_dir):
    print(f"{shp_dir} already exists, skipping fix + conversion.")
else:
    print(f"Fixing {kml_name} and converting to shapefile...")
    with fiona.open(kml_name) as input:
        # Write to shapefile
        with fiona.open(
            shp_dir, "w", "ESRI Shapefile", input.schema.copy(), input.crs # type: ignore
        ) as output:
            # Iterate through features
            for elem in input:
                if elem["geometry"] is None:
                    # We don't care about this feature
                    continue

                if elem["geometry"].type == "MultiLineString":
                    # Remove single-point LineStrings
                    out_coords = [
                        l for l in elem["geometry"].coordinates if len(l) >= 2
                    ]
                    if len(out_coords) > 0:
                        out_type = "MultiLineString"
                    else:
                        out_type = "LineString"
                    out_elem = {
                        "geometry": {"type": out_type, "coordinates": out_coords},
                        "properties": elem["properties"],
                    }
                    output.write(out_elem)

                elif elem["geometry"].type == "LineString":
                    if len(elem["geometry"].coordinates) >= 2:
                        output.write(elem)

# Load shapefile into GeoDataFrame

gdf = gpd.read_file(shp_dir + "/" + shp_dir + ".shp")
print(f"Loaded shape file ({len(gdf)} rows)!")

# Parse dates and add as column
gdf["Date"] = gdf["Descriptio"].str.extract(r"date=(.*)")[
    0
]  # get the part after 'date='
gdf["Date"] = pd.to_datetime(gdf["Date"], format="%d-%b-%Y %H:%M:%S")

# Split the coastline into segments based on latitude
min_lat, max_lat = gdf.total_bounds[1], gdf.total_bounds[3]
bins = np.arange(min_lat, max_lat + segment_height, segment_height)


# Assign each LineString a segment based on its centroid latitude
def assign_segment(line):
    return np.digitize(line.centroid.y, bins)


gdf["Segment"] = gdf["geometry"].apply(assign_segment)

print(len(gdf["Segment"].unique()), "segments defined!")

# Forward fill missing segments
print("Forward filling missing segments...")

# Combine geometries per Date + Segment
gdf_agg = gdf.groupby(["Date", "Segment"], as_index=False).agg(
    {"geometry": lambda x: unary_union(x)}
)

# Now pivot will work
pivot_gdf = gdf_agg.pivot(index="Date", columns="Segment", values="geometry")

# Sort by Date to ensure proper forward-fill
pivot_gdf = pivot_gdf.sort_index()

# Forward-fill missing geometries
pivot_gdf = pivot_gdf.ffill()

# Drop any initial rows that still have NaNs (before all segments appear)
pivot_gdf = pivot_gdf.dropna()

# Melt back into long GeoDataFrame format
filled_gdf = pivot_gdf.reset_index().melt(
    id_vars="Date", value_name="geometry", var_name="Segment"
)

# Convert back to GeoDataFrame
filled_gdf = gpd.GeoDataFrame(filled_gdf, geometry="geometry", crs=gdf.crs)

print(f"Total rows after filling: {len(filled_gdf)}")

# Save each segment to a separate file
print("Saving segments to files...")

# Ensure output directory exists and is empty
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Destroyed existing output directory: {output_dir}")

os.makedirs(output_dir)
print(f"Created directory: {output_dir}")

# Loop through each segment with a progress bar
unique_segments = filled_gdf["Segment"].unique()
for segment_id in tqdm(unique_segments, desc="Saving segments"):
    # Filter the main GeoDataFrame to get data only for the current segment
    segment_gdf = filled_gdf[filled_gdf["Segment"] == segment_id]

    # Define a clear filename for the output file
    # We use .gpkg (GeoPackage) as it's a modern, single-file format.
    output_filename = os.path.join(output_dir, f"segment_{segment_id}.gpkg")

    # Save the filtered GeoDataFrame to the file
    segment_gdf.to_file(output_filename, driver="GPKG")

print("All segments have been saved to the output directory! Goodbye :)")
