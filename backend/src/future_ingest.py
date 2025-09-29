"""
Takes in a `.csv` containing the output of our transformer, 
then returns future data about the coastline.
Written by Joshua Sheldon
On September 29, 2025
"""

from datetime import datetime
from typing import Dict

import pandas as pd
from shapely.affinity import translate
from shapely.geometry.linestring import LineString


def ingest_future_data(
    coastline_timestamp: datetime,
    latest_coastline: Dict[int, LineString],
    csv: str = "data/future/coastline_yearly_changes.csv",
) -> Dict[datetime, Dict[int, LineString]]:
    """
    Ingests future coastline change data and projects the coastline's position over time.

    Args:
        coastline_timestamp: The starting timestamp of the `latest_coastline`.
        latest_coastline: A dictionary mapping segment IDs to their initial LineString geometries.
        csv: The path to the CSV file containing yearly changes.

    Returns:
        A dictionary where keys are future datetimes and values are the
        corresponding projected coastline dictionaries for that year.
    """
    try:
        df = pd.read_csv(csv)
    except FileNotFoundError:
        print(f"Can't find CSV at path: {csv}!")
        return {}

    # --- Pandas-based Optimization ---
    # 1. Sort values by segment and year to ensure the cumulative sum is chronological.
    df.sort_values(by=["segment", "year"], inplace=True)

    # 2. Calculate the cumulative longitude change for each segment.
    # This is a vectorized operation that is much faster than a Python loop. It pre-calculates
    # the total shift from the origin for each segment at each given year.
    df["cumulative_lon_change"] = df.groupby("segment")["lon_change"].cumsum()

    # --- Data Processing ---
    future_coastlines: Dict[datetime, Dict[int, LineString]] = {}

    # Iterate through each unique year in the dataset to create a snapshot.
    for year in sorted(df["year"].unique()):
        # Create the new timestamp for the current year's snapshot.
        # We assume the change for a given year applies throughout that year.
        timestamp = coastline_timestamp.replace(year=(year - 1))

        # Start with a fresh copy of the original coastline for this year's calculation.
        # This is crucial for ensuring shifts are not compounded incorrectly.
        yearly_coastline_snapshot = latest_coastline.copy()

        # Find the latest cumulative change for each segment up to the current year.
        changes_up_to_year = df[df["year"] <= year]
        latest_change_per_segment = changes_up_to_year.drop_duplicates(
            subset="segment", keep="last"
        )

        # Apply the pre-calculated cumulative changes to the original LineStrings.
        for _, row in latest_change_per_segment.iterrows():
            segment_id = int(row["segment"])

            # Check if the segment from the CSV exists in our initial coastline data.
            if segment_id in yearly_coastline_snapshot:
                original_linestring = latest_coastline[segment_id]
                total_lon_shift = row["cumulative_lon_change"]

                # Apply the shift. A "left" shift corresponds to a negative change
                # in the x-coordinate (longitude).
                # We use shapely.affinity.translate, which is efficient for this operation.
                # The lat_change is ignored as per the prompt's specific instruction.
                shifted_linestring = translate(
                    original_linestring, xoff=-total_lon_shift
                )

                # Update the snapshot for this year with the new, shifted LineString.
                yearly_coastline_snapshot[segment_id] = shifted_linestring

        # Store the completed snapshot for the year in our results dictionary.
        future_coastlines[timestamp] = yearly_coastline_snapshot

    return future_coastlines
