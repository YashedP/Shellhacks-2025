import math
from typing import Dict, Optional

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
