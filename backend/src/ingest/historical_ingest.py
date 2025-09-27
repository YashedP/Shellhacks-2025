import os
import geopandas as gpd
from tqdm import tqdm
from typing import Dict, Union
from datetime import datetime
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from coastline_tree import CoastlineTreeNode
import matplotlib.pyplot as plt

def ingest_historical_data(dir: str = 'data/historical') -> Dict[datetime, Dict[int, LineString]]:
    """
    Given a directory of `.gpkg` files produced by the ETL pipeline,
    retuns a dictionary of timestamp to a dictionary of segment ID 
    to segment.
    """

    """
    {timestamp: {segment_id: geometry}
    """
    coastline_over_time = {}
    gpkg_files = [f for f in os.listdir(dir)]

    # Process every segment
    for gpkg_file in tqdm(gpkg_files, desc="Ingesting historical segments data"):
        segment_id = int(gpkg_file.split('segment_')[1].split('.')[0])
        df = gpd.read_file(os.path.join(dir, gpkg_file))

        # Drop 'Segment' column from df
        df = df.drop('Segment', axis=1)

        # Get dictionary from date to geometry
        data_dict = df.set_index('Date')['geometry'].to_dict()

        # Ditch pandas timestamp type
        data_dict = {key.to_pydatetime(): value for key, value in data_dict.items()}

        # Created sorted list of keys
        for timestamp, geometry in data_dict.items():
            if timestamp not in coastline_over_time:
                # Init sub-dict
                coastline_over_time[timestamp] = {}

            coastline_over_time[timestamp][segment_id] = geometry

    return coastline_over_time


data = ingest_historical_data()

for timestamp, coastline in data.items():
    node = CoastlineTreeNode(coastline)
    while (len(node.children) > 0):
        root_geometry = node.coastline.geometry
        gpd.GeoSeries([root_geometry], crs="EPSG:4326").plot()
        plt.show()
        node = node.children[0]
    root_geometry = node.coastline.geometry
    gpd.GeoSeries([root_geometry], crs="EPSG:4326").plot()
    plt.show()
    break
