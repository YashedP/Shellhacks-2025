"""
Takes in a folder of `.gpkg` files, each representing how a
segment of the coastline changes over time. Formats it
to be consumed by the `CoastlineMgr`.
Written by Yash Jani
On September 28, 2025.
"""

import os
from datetime import datetime
from typing import Dict
import numpy as np
import sys
import stats
import geopandas as gpd
import pandas as pd
from shapely.geometry.linestring import LineString
from shapely.geometry import Point
from tqdm import tqdm


def ingest_future_data(
    dir: str = "data/predictions",
) -> Dict[datetime, Dict[int, LineString]]:
    """
    Given a directory of `.gpkg` files produced by the prediction pipeline,
    returns a dictionary of timestamp to a dictionary of segment ID
    to segment for future predictions (2026-2100).
    """

    """
    {timestamp: {segment_id: geometry}
    """
    coastline_over_time = {}
    gpkg_files = [f for f in os.listdir(dir) if f.endswith('.gpkg') and 'rollout' in f]

    # Process every segment with progress bar
    for gpkg_file in tqdm(gpkg_files, desc="Ingesting future segments data"):
        segment_id = int(gpkg_file.split("segment_")[1].split("_rollout")[0])
        df = gpd.read_file(os.path.join(dir, gpkg_file))

        # Get dictionary from step to geometry
        data_dict = df.set_index("step")["geometry"].to_dict()

        # Convert step (which represents years 2026+) to datetime
        for step, geometry in data_dict.items():
            # Convert step to actual year (step 0 = 2026, step 1 = 2027, etc.)
            year = 2021 + step
            timestamp = datetime(year, 1, 1)  # Use January 1st of each year
            
            if timestamp not in coastline_over_time:
                # Init sub-dict
                coastline_over_time[timestamp] = {}

            coastline_over_time[timestamp][segment_id] = geometry

    return coastline_over_time


def generate_future_predictions_from_stats(
    stats_means: Dict[int, float],
    historical_dir: str = "data/historical",
    start_year: int = 2022,
    end_year: int = 2100
) -> Dict[datetime, Dict[int, LineString]]:
    """
    Generate future predictions by loading the latest historical dataframe files and adding statistical means to each point.
    
    Args:
        historical_dir: Directory containing historical .gpkg files
        stats_means: Dictionary mapping segment_id to mean latitude change
        start_year: Starting year for predictions (default 2022)
        end_year: Ending year for predictions (default 2100)
    
    Returns:
        Dictionary of timestamp to segment geometries for future years
    """
    # Load the latest 2021 data directly from historical files
    latest_2021_data = {}
    
    # Process all historical segment files to find the latest 2021 data
    gpkg_files = [f for f in os.listdir(historical_dir) if f.endswith('.gpkg')]
    
    for gpkg_file in tqdm(gpkg_files, desc="Loading latest 2021 historical data"):
        segment_id = int(gpkg_file.split("segment_")[1].split(".")[0])
        df = gpd.read_file(os.path.join(historical_dir, gpkg_file))
        
        # Drop 'Segment' column from df
        df = df.drop("Segment", axis=1)
        
        # Get the latest 2021 data for this segment
        df['Date'] = pd.to_datetime(df['Date'])
        df_2021 = df[df['Date'].dt.year == 2021]
        
        if not df_2021.empty:
            # Get the latest date in 2021 for this segment
            latest_date = df_2021['Date'].max()
            latest_geometry = df_2021[df_2021['Date'] == latest_date]['geometry'].iloc[0]
            latest_2021_data[segment_id] = latest_geometry
    
    future_predictions = {}
    
    # Generate predictions with progress bar
    for year in tqdm(range(start_year, end_year + 1), desc="Generating future predictions"):
        timestamp = datetime(year, 1, 1)
        future_predictions[timestamp] = {}
        
        for segment_id, geometry in latest_2021_data.items():
            if segment_id in stats_means:
                # Get the mean change for this segment
                mean_change = stats_means[segment_id]
                
                # Calculate how many years ahead we are from 2021
                years_ahead = year - 2021
                
                # Apply the mean change for each year
                total_change = mean_change * years_ahead
                
                # Deep copy the geometry and add mean change to latitude
                import copy
                new_geometry = copy.deepcopy(geometry)
                
                if hasattr(new_geometry, 'coords'):
                    # For LineString geometries - modify the deep copied geometry
                    coords_list = list(new_geometry.coords)
                    for i, coord in enumerate(coords_list):
                        new_lat = coord[1] + total_change  # Add change to latitude
                        coords_list[i] = (coord[0], new_lat)
                    
                    # Update the geometry with new coordinates
                    new_geometry = LineString(coords_list)
                    future_predictions[timestamp][segment_id] = new_geometry
                else:
                    # For other geometry types, use the deep copied geometry as is
                    future_predictions[timestamp][segment_id] = new_geometry
            else:
                # If no stats available for this segment, use original geometry
                future_predictions[timestamp][segment_id] = geometry
    
    return future_predictions


def compute_stats_means(predictions_dir: str = "data/predictions") -> Dict[int, float]:
    """
    Compute mean latitude changes for each segment from prediction data.
    
    Args:
        predictions_dir: Directory containing prediction .gpkg files
    
    Returns:
        Dictionary mapping segment_id to mean latitude change
    """
    stats_means = {}
    
    # Process all segments (1-250 based on stats.py) with progress bar
    for segment in tqdm(range(1, 251), desc="Computing stats for segments"):
        gpkg_path = os.path.join(predictions_dir, f"segment_{segment}_rollout.gpkg")
        
        if os.path.exists(gpkg_path):
            try:
                # Import the stats function
                from stats import compute_lat_change_stats
                
                # Compute stats for this segment
                stats = compute_lat_change_stats(gpkg_path, date_col="step", seg_col="segment")
                
                # Store the mean for this segment
                if "real_mean" in stats:
                    stats_means[segment] = stats["real_mean"]
                    
            except Exception as e:
                print(f"Error processing segment {segment}: {e}")
                # Use a default mean if processing fails
                stats_means[segment] = 0.0
    
    return stats_means
