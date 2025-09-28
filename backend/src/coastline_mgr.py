"""
Manages past and future coastlines.
Assembles data structure that assist in retrieval.
Written by Joshua Sheldon
On September 28, 2025
"""

from datetime import datetime
from typing import Dict, Tuple

from shapely.geometry.linestring import LineString
from tqdm import tqdm

from coastline_tree import CoastlineTreeNode
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


CoastlineMgr()
