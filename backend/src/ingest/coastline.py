from typing import Union, Tuple, List
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import unary_union

class Coastline():
    """
    A data structure which represents a coastline.
    Exposes a bounding box for the coastline, and a list of 
    lines, which are each composed of (lat, lon) coordinates.
    """
    def __init__(self, data: Union[List["Coastline"], LineString, MultiLineString]):
        if not isinstance(data, List):
            self.geometry = data
        else:
            if len(data) == 1:
                self.geometry = data[0].geometry
            else:
                geometries = [coastline.geometry for coastline in data]
                self.geometry = unary_union(geometries)
        
        self.bounding_box = self.geometry.envelope
        self.lines: List[List[Tuple[float, float]]] = []

        # Instantiate lines
        if isinstance(self.geometry, LineString):
            line = []
            for coord in self.geometry.coords:
                line.append((coord[0], coord[1]))
            self.lines.append(line)
        elif isinstance(self.geometry, MultiLineString):
            for line in self.geometry.geoms:
                line_list = []
                for coord in line.coords:
                    line_list.append((coord[0], coord[1]))
                self.lines.append(line_list)
        else:
            print('Unknown geometry type: ', type(self.geometry))
            self.lines = []