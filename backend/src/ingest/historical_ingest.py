import os
import geopandas as gpd
from tqdm import tqdm
from typing import Dict, Tuple, List
from datetime import datetime
from shapely.geometry.base import BaseGeometry

def ingest_historical_data(dir: str = '../../data/historical') -> Dict[int, Tuple[List[datetime], Dict[datetime, BaseGeometry]]]:
    """
    Given a directory of `.gpkg` files produced by the ETL pipeline,
    retuns a dictionary of segment ID to segments, where a segment is 
    a tuple. The first element of the tuple is a sorted list of 
    timestamps. The second element of the tuple is a dictionary from 
    timestamp to geometry (the shape of that segment of the coastline 
    at that time). The geometry may either be `LineString` or 
    `MultiLineString`.
    """

    """
    {segment_id: (sorted_timestamps, {timestamp: geometry})}
    """
    segments = {}
    gpkg_files = [f for f in os.listdir(dir)]

    # Process every segment
    for gpkg_file in tqdm(gpkg_files, desc="Ingesting historical segments data"):
        segment_id = int(gpkg_file.split('segment_')[1].split('.')[0])
        df = gpd.read_file(os.path.join(dir, gpkg_file))

        # Drop 'Segment' column from df
        df = df.drop('Segment', axis=1)

        # Drop duplicate geometries
        df = df.drop_duplicates(subset='geometry').reset_index(drop=True)

        # Get dictionary from date to geometry
        data_dict = df.set_index('Date')['geometry'].to_dict()

        # Ditch pandas timestamp type
        data_dict = {key.to_pydatetime(): value for key, value in data_dict.items()}

        # Created sorted list of keys
        sorted_timestamps = list(data_dict.keys())
        sorted_timestamps.sort()

        # Add to segments dict
        segments[segment_id] = (sorted_timestamps, data_dict)

    return segments

# Get average number of vertices in geometry
segments = ingest_historical_data()

avg_length = []

for segment in segments:
    dictionary = segments[segment][1]
    for key, value in dictionary.items():
        if value.geom_type == 'LineString':
            avg_length.append(len(value.coords))
        elif value.geom_type == 'MultiLineString':
            avg_length.append(sum(len(line.coords) for line in value.geoms))
        else:
            print('UHOH')

print(sum(avg_length) / len(avg_length))