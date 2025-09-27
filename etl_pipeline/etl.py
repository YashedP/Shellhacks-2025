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
from shapely.geometry import Point, LineString
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----- CONFIG -----

kmz_url = "https://cmgds.marine.usgs.gov/data-releases/media/2022/10.5066-P9BQQTCI/555bff79a2e14785a1347c7ad285c8a0/SatelliteDerivedShorelines_FL.kmz"
kmz_name = "SatelliteDerivedShorelines_FL.kmz"
kml_name = "CoastSat_shorelines_FL.kml"
shp_dir = "CoastSat_shorelines_FL"
segment_height = 0.02
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

# Combine geometries per Date + Segment
gdf_agg = gdf.groupby(["Date", "Segment"], as_index=False).agg(
    {"geometry": lambda x: unary_union(x)}
)

# We've run into a lot of issues with MultiLineStrings in the backend
# Let's connect all MultiLineStrings into SingleLineStrings
def connect_multilinestring_by_latitude(geom):
    """
    If a geometry is a MultiLineString, connects its component LineStrings
    from north to south to form a single LineString. It connects the south
    end of the northern line to the north end of the southern line.
    """
    # If it's not a MultiLineString or is empty, return it as-is.
    if geom.geom_type != 'MultiLineString' or geom.is_empty:
        return geom

    lines = [line for line in geom.geoms if not line.is_empty and len(line.coords) > 1]

    # Handle simple cases
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]

    # 1. Calculate the average latitude for each line
    lines_with_lat = []
    for line in lines:
        avg_lat = np.mean([pt[1] for pt in line.coords])
        lines_with_lat.append((avg_lat, line))

    # 2. Sort lines from north to south (descending latitude)
    lines_with_lat.sort(key=lambda x: x[0], reverse=True)
    sorted_lines = [line for lat, line in lines_with_lat]

    # 3. Iteratively connect the lines using the new logic
    connected_line = sorted_lines[0]
    for line_to_add in sorted_lines[1:]:
        coords1 = list(connected_line.coords)
        coords2 = list(line_to_add.coords)

        # Orient the current line (coords1) so its southernmost point is at the end.
        # If the starting point's latitude is less than the ending point's, it's the southern point.
        if coords1[0][1] < coords1[-1][1]:
            coords1.reverse() # Reverse so the southern point is the last element

        # Orient the next line (coords2) so its northernmost point is at the beginning.
        # If the ending point's latitude is greater than the starting point's, it's the northern point.
        if coords2[-1][1] > coords2[0][1]:
            coords2.reverse() # Reverse so the northern point is the first element

        # Combine the coordinate lists and create the new, longer LineString
        connected_line = LineString(coords1 + coords2)
        
    return connected_line

# Apply the function
print("Connecting MultiLineString geometries...")
tqdm.pandas(desc="Connecting lines")
gdf_agg['geometry'] = gdf_agg['geometry'].progress_apply(connect_multilinestring_by_latitude)

# Drop any rows with empty geometries after processing
gdf_agg.dropna(subset=['geometry'], inplace=True)

print("Forward filling...")

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
