# Standard libraries
import math

# Third-party libraries
import numpy as np
import scipy.stats
import shapely.geometry
import shapely.ops


# Local libraries
from pyscoring.metrics.metric import Metric


def compute_distance(coords_a, coords_b):
    point_A = shapely.geometry.Point(coords_a)
    point_B = shapely.geometry.Point(coords_b)

    return point_A.distance(point_B)


class IntrinsicMetrics(Metric):
    """Class to compute various statistics on a set of features.

    Currently, the class aggregates for each feature/polygon:
        * the angle between two adjacent edges (in degrees)
        * the area (in meters)
        * the length for each edge (in meters)

    Finally, the user may want to compute some statistics about the different parameter.
    The compute method allows the user to compute min/max/mean/variance values over each distribution.

    """

    def __init__(
        self,
        crs="EPSG:4326",
        trim_invalid_geometry=False,
        autocorrect_invalid_geometry=False,
    ):

        self.trim_invalid_geometry = trim_invalid_geometry
        self.autocorrect_invalid_geometry = autocorrect_invalid_geometry
        self.crs = crs

        self._init_values()

    def _init_values(self):
        # Init attributes
        self.angles = []
        self.areas = []
        self.lengths = []
        self.holes = 0

        self.MLA = 0
        self.MSP = 0
        self.NE = 0
        self.NS = 0
        self.NSP = 0
        self.NTP = 0

    def update(self, features):
        """Accumulate features and compute some geometrical information (area, edges length, etc.)

        Args:
            features (list): List of features to extract geometrical metrics

        """

        detections = features
        # TODO : maybe test the type of the detections to avoid bugs when something else than a Polygon is given

        for feature in detections:

            # Compute areas
            area = self._compute_area(feature, self.crs)
            self.areas.append(area)

            # Compute vertices lengths
            lengths = self._compute_lengths(feature)
            self.lengths.extend(lengths)

            self.holes += len(feature.interiors)

    def compute(self):
        """Compute statistics (min, mean, median, max, sum).

        Returns:
            tuple: statistics over each parameter, a.k.a angle, area and lengths

        """
        if len(self.lengths) != 0 and len(self.areas) != 0:
            # Areas
            areas_stats = scipy.stats.describe(np.array(self.areas))

            # Lengths
            lengths_stats = scipy.stats.describe(np.array(self.lengths))

            self.NE = areas_stats.nobs
            self.NS = lengths_stats.nobs

            self.NSP = self.NS / self.NE
            self.NTP = self.holes / self.NE

            self.MLA = lengths_stats.mean
            self.MSP = areas_stats.mean

            return self.MLA, self.MSP, self.NE, self.NS, self.NSP, self.NTP
        else:
            return self.MLA, self.MSP, self.NE, self.NS, self.NSP, self.NTP

    def reset(self):
        """This method automatically resets the metric state variables to their default value."""
        self._init_values()

    @staticmethod
    def _compute_angles(geometry):
        """Compute polygon angles (between polygon edges)."""
        angles = []
        points = geometry.exterior.coords
        for index in range(1, len(points) - 1):

            prev = (index - 1) % len(points)
            cur = index % len(points)
            next = (index + 1) % len(points)

            vector_1 = [
                points[cur][0] - points[prev][0],
                points[cur][1] - points[prev][1],
            ]
            vector_2 = [
                points[next][0] - points[cur][0],
                points[next][1] - points[cur][1],
            ]

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

            dot_product = np.dot(unit_vector_1, unit_vector_2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = math.degrees(np.arccos(dot_product))
            angles.append(angle)

        return angles

    @staticmethod
    def _compute_area(geometry, crs):
        """Compute polygon areas.
        Should not be any problem in projected coordinates"""

        return geometry.area

    @staticmethod
    def _compute_lengths(geometry):
        """Compute polygon lengths."""
        lengths = []
        if geometry.geom_type == "MultiPolygon":
            print(geometry)
        points = geometry.exterior.coords
        for i in range(len(points) - 1):
            lengths.append(compute_distance(points[i], points[i + 1]))
        return lengths
