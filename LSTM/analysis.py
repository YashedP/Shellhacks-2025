import sys
import json
import geopandas as gpd
import os


files = [
    os.path.join("./output_segments", f)
    for f in os.listdir("./output_segments")
    if f.endswith(".gpkg")
]

def load_coastline_data(file: str) -> gpd.GeoDataFrame:
    """Load coastline data from files"""
    return gpd.read_file(file)


# def test_for_one_segment():
#     df = load_coastline_data(files[0])

#     line_string_num_points = []

#     for i in range(0, 30):
#         geom = df.iloc[i].geometry
#         num_points = 0

#         for j, line in enumerate(geom.geoms):
#             num_points += len(list(line.coords))

#         if num_points not in num_coordinate_points:
#             num_coordinate_points[num_points] = 1
#         else:
#             num_coordinate_points[num_points] += 1


#     print(f"dict of number of coordinate points: {num_coordinate_points}")

#     import json

#     with open("num_coordinate_points.json", "w") as f:
#         json.dump(num_coordinate_points, f, indent=2)

def test_for_all_segments():
    for file in files:
        df = load_coastline_data(file)
        

if __name__ == "__main__":
    # test_for_one_segment()
    test_for_all_segments()