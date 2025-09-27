from typing import Dict, Union
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.linestring import LineString
import math
from coastline import Coastline

class CoastlineTreeNode():
    """
    A data structure which represents increasingly high 
    fidelity segments of a coastline at a certain point 
    in time.
    """

    def __init__(self, segments: Dict[int, Union[LineString, MultiLineString]]):
        # Each node should have 4 children, so split segments
        # into groups or dub this node a leaf node
        child_groups = []
        if len(segments) > 1:
            children_per_group = math.ceil(len(segments) / 4)
            children_added = 0
            
            # Distribute segments among children
            for segment_id, geometry in segments.items():
                group_index = math.floor(children_added / children_per_group)

                if len(child_groups) == group_index:
                    # Add list if DNE
                    child_groups.append({})
                
                child_groups[group_index][segment_id] = geometry
                children_added += 1

        self.children = [CoastlineTreeNode(child_group) for child_group in child_groups]

        # Define coastline based on children if we have them,
        # or just self if leaf node
        if len(self.children) < 1:
            self.coastline = Coastline(list(segments.values())[0])
        else:
            # Assemble coastline from those of all children
            child_coastlines = [child.coastline for child in self.children]
            self.coastline = Coastline(child_coastlines)

