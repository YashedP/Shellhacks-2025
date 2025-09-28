"""
A tree, where each non-leaf node has 4 children, to assist in
merging many segments into larger and larger portions of the
coastline. This tree allows a progressive elaboration in the
frontend, where the map gets more detailed as you zoom in
without loading all vertices from the outset.
Written by Joshua Sheldon and Yash Jani
On September 28, 2025
"""

import math
from typing import Dict, List, Tuple

from shapely.geometry import Polygon, box
from shapely.geometry.linestring import LineString

from coastline import Coastline


class CoastlineTreeNode:
    """
    A data structure which represents increasingly high
    fidelity segments of a coastline at a certain point
    in time.
    """

    def _assemble_lines(self, segments: Dict[int, LineString]):
        """
        Assemble list of lines IN ORDER from segments.
        """
        lines = []
        for segment_id in self.segments:
            lines.append(segments[segment_id])
        return lines

    @staticmethod
    def _calculate_bounding_box(lines: List[LineString]) -> Polygon:
        """
        Efficiently calculates the bounding box by iterating over the
        bounds of individual LineStrings without merging them.
        """
        if not lines:
            return Polygon()

        # Initialize with the bounds of the first segment
        min_x, min_y, max_x, max_y = lines[0].bounds

        # Expand the bounds by checking all other segments
        for line in lines[1:]:
            l_min_x, l_min_y, l_max_x, l_max_y = line.bounds
            min_x = min(min_x, l_min_x)
            min_y = min(min_y, l_min_y)
            max_x = max(max_x, l_max_x)
            max_y = max(max_y, l_max_y)

        # Create the bounding box polygon from the final coordinates
        return box(min_x, min_y, max_x, max_y)

    def __init__(self, segments: Dict[int, LineString]):
        # Each node should have 4 children, so split segments
        # into groups or dub this node a leaf node
        child_groups = []
        if len(segments) > 1:
            children_per_group = math.ceil(len(segments) / 4)
            children_added = 0

            # Distribute segments among children
            for segment_id, geometry in sorted(segments.items()):
                group_index = min(math.floor(children_added / children_per_group), 3)

                if len(child_groups) == group_index:
                    # Add list if DNE
                    child_groups.append({})

                child_groups[group_index][segment_id] = geometry
                children_added += 1

        self.segments = sorted(list(segments.keys()))
        self.children = [CoastlineTreeNode(child_group) for child_group in child_groups]

        self.bounding_box: Polygon = self._calculate_bounding_box(
            self._assemble_lines(segments)
        )
        self.depth = -1

    def get_coastline(self, segments: Dict[int, LineString]):
        # Create object
        return Coastline(self._assemble_lines(segments), self.depth)
    
    def is_fully_contained_in(self, query_bounds: Tuple[float, float, float, float]) -> bool:
        if self.bounding_box is None:
            return False
        
        # Get bounding box coordinates
        minx, miny, maxx, maxy = self.bounding_box.bounds
        
        # Check if node's bounding box is fully contained within query bounds
        query_min_lon, query_min_lat, query_max_lon, query_max_lat = query_bounds
        
        return (query_min_lon <= minx and query_min_lat <= miny and 
                query_max_lon >= maxx and query_max_lat >= maxy)

    def get_points_in_bounds(self, segments: Dict[int, LineString], 
                            query_bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        points = []
        
        # If this node is fully contained, return all its points
        if self.is_fully_contained_in(query_bounds):
            coastline = self.get_coastline(segments)
            points.extend(coastline.line)
        else:
            # Otherwise, check children
            for child in self.children:
                points.extend(child.get_points_in_bounds(segments, query_bounds))
        
        return points
