from typing import Union, Tuple, List
from shapely.geometry.linestring import LineString
import numpy as np

class Coastline():
    """
    A data structure which represents a coastline.
    Exposes a bounding box for the coastline, and a list of 
    lines, which are each composed of (lat, lon) coordinates.
    """

    @staticmethod
    def _connect_geometry(lines: List[LineString]) -> LineString:
        """
        Connects a list of LineStrings into a single, continuous LineString,
        ordered from north to south.
        """
        # Filter out any invalid or empty geometries from the input list
        valid_lines = [line for line in lines if isinstance(line, LineString) and not line.is_empty and len(line.coords) > 1]

        # Handle simple cases after filtering
        if not valid_lines:
            raise Exception("No valid lines when connecting geometry!")
        if len(valid_lines) == 1:
            return valid_lines[0]

        # 1. Calculate the average latitude for each line to determine its general position
        lines_with_lat = []
        for line in valid_lines:
            avg_lat = np.mean([pt[1] for pt in line.coords])
            lines_with_lat.append((avg_lat, line))

        # 2. Sort lines from north to south (descending latitude)
        lines_with_lat.sort(key=lambda x: x[0], reverse=True)
        sorted_lines = [line for _, line in lines_with_lat]

        # 3. Iteratively connect the sorted lines
        connected_line = sorted_lines[0]
        for line_to_add in sorted_lines[1:]:
            coords1 = list(connected_line.coords)
            coords2 = list(line_to_add.coords)

            # Orient the current connected line so its southernmost point is at the end
            if coords1[0][1] < coords1[-1][1]:
                coords1.reverse()

            # Orient the new line to be added so its northernmost point is at the beginning
            if coords2[-1][1] > coords2[0][1]:
                coords2.reverse()

            # Combine the coordinates to form the new, longer LineString
            connected_line = LineString(coords1 + coords2)
            
        return connected_line

    def __init__(self, data: Union[List["Coastline"], LineString]):
        if not isinstance(data, List):
            self.geometry = data
        else:
            if len(data) == 1:
                self.geometry = data[0].geometry
            else:
                lines_to_process = [coastline.geometry for coastline in data]
                self.geometry = self._connect_geometry(lines_to_process)
        
        self.bounding_box = self.geometry.envelope
        self.line: List[Tuple[float, float]] = []

        # Instantiate lines
        if isinstance(self.geometry, LineString):
            for coord in self.geometry.coords:
                self.line.append((coord[0], coord[1]))
        else:
            print('Unknown geometry type: ', type(self.geometry))
        