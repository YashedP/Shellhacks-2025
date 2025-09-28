import math
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.geometry.linestring import LineString

from coastline import Coastline


class CoastlineTreeNode:
    """
    A data structure which represents increasingly high
    fidelity segments of a coastline at a certain point
    in time.
    """

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
        self.bounding_box: Optional[Polygon] = None

    def get_coastline(self, segments: Dict[int, LineString]):
        # Assemble lines
        lines = []
        for segment_id in self.segments:
            lines.append(segments[segment_id])

        # Create object
        return Coastline(lines)

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
