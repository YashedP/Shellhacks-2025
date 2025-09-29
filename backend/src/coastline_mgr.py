"""
Manages past and future coastlines.
Assembles data structure that assist in retrieval.
Written by Joshua Sheldon and Yash Jani
On September 28, 2025
"""

from datetime import datetime
from typing import Dict, List, Tuple

from shapely.geometry.linestring import LineString
from tqdm import tqdm

from coastline_tree import CoastlineTreeNode
from future_ingest import ingest_future_data
from historical_ingest import ingest_historical_data


class CoastlineMgr:
    def _populate_depth(self, node: CoastlineTreeNode, depth: int):
        # Set depth
        node.depth = depth

        # Do some for children
        for child_node in node.children:
            self._populate_depth(child_node, depth + 1)

    def __init__(self):
        self.coastlines: Dict[
            datetime, Tuple[CoastlineTreeNode, Dict[int, LineString]]
        ] = {}

        historical_coastlines = ingest_historical_data()

        for timestamp, segments in tqdm(
            historical_coastlines.items(), desc="Processing historical coastlines"
        ):
            root = CoastlineTreeNode(segments)
            self._populate_depth(root, 0)
            self.coastlines[timestamp] = (root, segments)

        latest_timestamp = max(self.coastlines.keys())

        future_coastlines = ingest_future_data(
            latest_timestamp, self.coastlines[latest_timestamp][1]
        )

        for timestamp, segments in tqdm(
            future_coastlines.items(), "Processing future coastlines"
        ):
            root = CoastlineTreeNode(segments)
            self._populate_depth(root, 0)
            self.coastlines[timestamp] = (root, segments)

    def get_coastline_points(
        self, timestamp: datetime, query_bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Get coastline points for a specific timestamp within the given query bounds.
        Query bounds format: (min_lon, min_lat, max_lon, max_lat)
        """
        if timestamp not in self.coastlines:
            return []

        root, segments = self.coastlines[timestamp]
        return root.get_points_in_bounds(segments, query_bounds)
