from datetime import datetime
from typing import Dict, List, Tuple

from shapely.geometry.linestring import LineString
from tqdm import tqdm

from coastline_tree import CoastlineTreeNode
from historical_ingest import ingest_historical_data


class CoastlineMgr:
    def _populate_bounding_boxes(
        self, node: CoastlineTreeNode, segments: Dict[int, LineString]
    ):
        # Set bounding box attribute
        coastline = node.get_coastline(segments)
        node.bounding_box = coastline.bounding_box

        # Do some for children
        for child_node in node.children:
            self._populate_bounding_boxes(child_node, segments)

    def __init__(self):
        self.coastlines: Dict[
            datetime, Tuple[CoastlineTreeNode, Dict[int, LineString]]
        ] = {}

        historical_coastlines = ingest_historical_data()

        for timestamp, segments in tqdm(
            historical_coastlines.items(), desc="Processing historical coastlines"
        ):
            root = CoastlineTreeNode(segments)
            self._populate_bounding_boxes(root, segments)
            self.coastlines[timestamp] = (root, segments)

    def get_coastline_points(self, timestamp: datetime, 
                           query_bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """
        Get coastline points for a specific timestamp within the given query bounds.
        Query bounds format: (min_lon, min_lat, max_lon, max_lat)
        """
        if timestamp not in self.coastlines:
            return []
        
        root, segments = self.coastlines[timestamp]
        return root.get_points_in_bounds(segments, query_bounds)

