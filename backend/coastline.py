"""
A data structure which represents a coastline. Typically
used to merge multiple segments into one, contiguous line.
Written by Joshua Sheldon
On September 28, 2025.
"""

from typing import List, Tuple

from shapely import simplify
from shapely.geometry.linestring import LineString

from hardcoded_tolerances import simplify_tolerances


class Coastline:
    """
    A data structure which represents a coastline.
    Exposes a bounding box for the coastline, and a list of
    lines, which are each composed of (lat, lon) coordinates.
    """

    @staticmethod
    def _assemble_line(lines: List[LineString]) -> LineString:
        """
        Assembles a list of LineString objects into a single LineString.
        """
        all_coords = []
        for line in lines:
            # Ensure the line is not empty before extending coordinates
            if not line.is_empty:
                all_coords.extend(list(line.coords))

        # If there are no coordinates, return an empty LineString
        if not all_coords:
            return LineString()

        # Remove consecutive duplicate points which can occur at segment junctions
        unique_coords = [all_coords[0]]
        for point in all_coords[1:]:
            if point != unique_coords[-1]:
                unique_coords.append(point)

        # A valid LineString requires at least two distinct points
        if len(unique_coords) < 2:
            return LineString()

        return LineString(unique_coords)

    def __init__(self, lines: List[LineString], depth: int):
        # This is a sorted list, meaning line 1 connects to line 2,
        # line 2 connects to line 3, so on and so forth
        self.geometry = self._assemble_line(lines)

        # Simplify line if there's a valid depth for it
        if depth in simplify_tolerances:
            self.geometry = simplify(self.geometry, simplify_tolerances[depth], True)

        self.bounding_box = self.geometry.envelope
        self.line: List[Tuple[float, float]] = []

        # Instantiate lines
        if isinstance(self.geometry, LineString):
            for coord in self.geometry.coords:
                self.line.append((coord[0], coord[1]))
        else:
            print("Unknown geometry type: ", type(self.geometry))
