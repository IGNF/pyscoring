"""Implement the public interface to compute 4 extrinsic geometrical metrics of edges from a set of detections and ground truths.

If one wants to integrate the module into a framework to use it as a validation metric, the
:class:`~ExtrinsicGeomEdgeMetric` class described below should be wrappped accordingly to follow the framework
convention.
"""

import numpy as np
from collections import defaultdict
from shapely.ops import unary_union
import shapely.geometry as geom
import math
from fiona import open


class EdgesMetric:
    r"""Implement an API to compute the Extrinsic Geometric edges metrics.
    Those metrics are defined as the followings:
        * PCont (meters), "Précision de contour" :  mean distance from a dataset to evaluate RE/detections to a groundtruths dataset GT.
        * RCont (meters), "Rappel de contour" : mean distance from a dataset of groundtruths GT  to a dataset to evaluate RE/detections.
        * POri (degrees), "Précision d'orientation" : mean angle between the edges of a dataset to evaluate RE/detections and the edges of GT.
        * ROri (degrees), "Rappel d'orientaiton" : mean angle between the edges of a dataset GT to the edges of RE/detections.

    It gives three methods:

        * :meth:`update(detections, ground_truths) <update>` which accumulates distances and angles over examples
        * :meth:`compute` which computes PCont, RCont, POri, ROri per label from accumulated values
        * :meth:`reset` which resets accumulated values to their initial values to start computations from scratch

    """

    def __init__(
        self,
        pixel_size=0.5,
        match_algorithm=None,
        threshold=0.5,
        matched=True,
        mode=0,
        savePoints=False,
        union=False,
    ):
        r"""

        Args:
        match_algorithm (str): Optional, default to 'coco'. 'xview' or 'coco' to choose the matching algorithm (c.f.
            :ref:`match`) or 'non-unitary' to use non-unitary matching.
        threshold (float): Optional, default to 0.5. Similarity threshold for which we consider a valid
            match between detection and ground truth.
        match_engine (:class:`~map_metric_api.match_detections.MatchEngineBase`): Optional, default to
            :class:`~map_metric_api.match_detections.MatchEngineIoU`. If provided matching will be done using the
            provided ``match_engine`` instead of the default one. Note that the ``threshold`` and ``match_algorithm``
            provided parameters will be overridden by those provided in the ``match_engine``.
        mode (int): if 0 the calculation will be on both detection and gt, with 1 only for detections and with 2 only for gt
        savePoints (Bool) : If True save the points sampling the datasets.
        """
        self.pixel_size = pixel_size
        self.match_algorithm = match_algorithm or "ioma"
        self.threshold = threshold
        self.matched = matched

        # self.trim_invalid_geometry = trim_invalid_geometry
        # self.autocorrect_invalid_geometry = autocorrect_invalid_geometry

        # if match_engine is not None and (threshold is not None or match_algorithm is not None):
        #    warnings.warn('In the future match_engine will be made incompatible with threshold and match_algorithm. '
        #                  'Providing both will raise a ValueError.', FutureWarning)

        # self.match_engine = match_engine or MatchEngineIoMA(threshold, match_algorithm)
        self.mode = mode
        self.svPts = savePoints
        self.union = union
        self._init_values()

    def _init_values(self):
        self._number_of_dataset = 0

        self._ground_truth_labels = set()

        # self.distPrecision = defaultdict(float)
        # self.distRappel = defaultdict(float)
        # self.slopePrecision = defaultdict(float)
        # self.slopeRappel = defaultdict(float)
        self.distPrecision = 0.0
        self.distRappel = 0.0
        self.maxDistP = 0.0
        self.maxDistR = 0.0

        self.slopePrecision = 0.0
        self.slopeRappel = 0.0
        self.maxAngleP = 0.0
        self.maxAngleR = 0.0

        self._numberDmatched = 0
        self._numberGTmatched = 0

        self._pc = 0.0  # PCont
        self._rc = 0.0  # RCont
        self._po = 0.0  # POri
        self._ro = 0.0  # ROri

    def update(self, detections, ground_truths, mode=None):
        """
        Accumulates metrics for new detections and ground truths

        :param detections: list of shapely.Polygon, the new detections to evaluate
        :param ground_truths: list of shapely.Polygon, the new ground truths to which we compare the detections
        :param mode:
        """
        if mode is None:
            mode = self.mode

        detections = detections
        ground_truths = ground_truths

        if len(detections) == 0 or len(ground_truths) == 0:
            return

        if self.union:
            detections = self.merge(detections)
            ground_truths = self.merge(ground_truths)

        polygon_detect = detections
        polygon_gt = ground_truths

        if self.matched:  # TODO not matched !
            match_matrix = np.ones((len(detections), len(ground_truths)))

        # Sampling the detections
        sampled_detection = []
        slope_detection = []
        total_notccw = 0

        for polygon in polygon_detect:
            linRing = geom.LinearRing(polygon.exterior)

            if not linRing.is_ccw:  # test the outside line
                total_notccw += 1

            sampled_polygon, slopes = self._sampling(polygon, self.pixel_size)

            sampled_detection.append(sampled_polygon)
            slope_detection.append(slopes)

        # Sampling the gt
        sampled_gt = []
        slope_gt = []
        total_notccwgt = 0
        for polygon in polygon_gt:
            linRing = geom.LinearRing(polygon.exterior)
            if not linRing.is_ccw:
                total_notccwgt += 1

            sampled_polygon, slopes = self._sampling(polygon, self.pixel_size)

            sampled_gt.append(sampled_polygon)
            slope_gt.append(slopes)

        # for each polygon in the dataset of detections : compute the means with the GT
        if mode == 0 or mode == 1:
            (
                mean_dist_precision,
                mean_slope_precision,
                max_dist_p,
                max_angle_p,
                index_detect,
                nbDmatched,
            ) = distance(
                [sampled_detection, slope_detection],
                [sampled_gt, slope_gt],
                match_matrix,
                self.svPts,
            )
            mean = [0, 0]
            for i in range(index_detect):
                if mean_dist_precision[i] != -1:
                    mean[0] += mean_dist_precision[i]
                    mean[1] += mean_slope_precision[i]

                # if not self.distPrecision[gt_label]:
                #     self.distPrecision[gt_label] = 0
                #     self.slopePrecision[gt_label] = 0
                #
                # self.distPrecision[gt_label] = ((self.distPrecision[gt_label]*self._number_of_dataset) \
                #                                             + (mean[0] / index_detect)) / (self._number_of_dataset+1)
                #
                # self.slopePrecision[gt_label] = ((self.slopePrecision[gt_label] * self._number_of_dataset) \
                #                                 + (mean[1] / index_detect)) / (self._number_of_dataset + 1)

                self.distPrecision = (
                    self.distPrecision * self._number_of_dataset
                    + (mean[0] / index_detect)
                ) / (self._number_of_dataset + 1)
                self.slopePrecision = (
                    (self.slopePrecision * self._number_of_dataset)
                    + (mean[1] / index_detect)
                ) / (self._number_of_dataset + 1)

                self.maxDistP = (
                    max_dist_p[i] if max_dist_p[i] > self.maxDistP else self.maxDistP
                )
                self.maxAngleP = (
                    max_angle_p[i]
                    if max_angle_p[i] > self.maxAngleP
                    else self.maxAngleP
                )

            self._numberDmatched += nbDmatched

        # for each polygon in the dataset of GT : compute the means with the detections
        if mode == 0 or mode == 2:
            (
                mean_dist_rappel,
                mean_slope_rappel,
                max_dist_r,
                max_angle_r,
                index_gt,
                nbGTmatched,
            ) = distance(
                [sampled_gt, slope_gt],
                [sampled_detection, slope_detection],
                match_matrix.T,
                False,
            )

            mean = [0, 0]
            for i in range(index_gt):
                if mean_dist_rappel[i] != -1:
                    mean[0] += mean_dist_rappel[i]
                    mean[1] += mean_slope_rappel[i]

                # if not self.distRappel[gt_label]:
                #     self.distRappel[gt_label] = 0
                #     self.slopeRappel[gt_label] = 0
                #
                # self.distRappel[gt_label] = ((self.distRappel[gt_label] * self._number_of_dataset) \
                #                                 + (mean[0] / index_gt)) / (self._number_of_dataset + 1)
                # self.slopeRappel[gt_label] = ((self.slopeRappel[gt_label] * self._number_of_dataset) \
                #                             + (mean[1] / index_gt)) / (self._number_of_dataset + 1)

                self.distRappel = (
                    self.distRappel * self._number_of_dataset + (mean[0] / index_gt)
                ) / (self._number_of_dataset + 1)
                self.slopeRappel = (
                    (self.slopeRappel * self._number_of_dataset) + (mean[1] / index_gt)
                ) / (self._number_of_dataset + 1)
                self.maxDistR = (
                    max_dist_r[i] if max_dist_r[i] > self.maxDistR else self.maxDistR
                )
                self.maxAngleR = (
                    max_angle_r[i]
                    if max_angle_r[i] > self.maxAngleR
                    else self.maxAngleR
                )

            self._numberGTmatched += nbGTmatched

        self._number_of_dataset += 1

    def compute(self):
        # sumpc = 0
        # for label in self.distPrecision:
        #     sumpc += self.distPrecision[label]
        # if len(self.distPrecision) != 0 :
        #     self._pc = sumpc / len(self.distPrecision)
        #
        # sumrc = 0
        # for label in self.distRappel:
        #     sumrc += self.distRappel[label]
        # if len(self.distRappel) != 0 :
        #     self._rc = sumrc / len(self.distRappel)
        #
        # sumpo = 0
        # for label in self.slopePrecision:
        #     sumpo += self.slopePrecision[label]
        # if len(self.slopePrecision) != 0 :
        #     self._po = sumpo / len(self.slopePrecision)
        #
        # sumro = 0
        # for label in self.slopeRappel:
        #     sumro += self.slopeRappel[label]
        # if len(self.slopeRappel) != 0 :
        #     self._ro = sumro / len(self.slopeRappel)

        return (
            self.distPrecision,
            self.distRappel,
            self.maxDistP,
            self.maxDistR,
            self.slopePrecision,
            self.slopeRappel,
            self.maxAngleP,
            self.maxAngleR,
            self._numberDmatched,
            self._numberGTmatched,
        )

    def reset(self):
        self._init_values()

    @staticmethod
    def _empty_array():
        return np.array([])

    def _sampling(self, polygon, rate):
        sample = [[]]
        slope = [[]]
        exterior_c = polygon.exterior.coords
        for index in range(len(exterior_c)):
            if index != len(exterior_c) - 1:
                edge = geom.LineString([exterior_c[index], exterior_c[index + 1]])
                origins = edge.coords
                while edge:
                    points, edge = self.cut(edge, rate)
                    sample[0].append(points[0])
                    slope[0].append(origins)
            elif (
                exterior_c[index][0] != exterior_c[0][0]
                and exterior_c[index][1] != exterior_c[0][1]
            ):
                edge = geom.LineString([exterior_c[index], exterior_c[0]])
                origins = edge.coords
                while edge:
                    points, edge = self.cut(edge, rate)
                    sample[0].append(points[0])
                    sample[0].append(points[1])
                    slope[0].append(origins)
                    slope[0].append(origins)

        for hole in polygon.interiors:
            hole_c = hole.coords
            sample.append([])
            for index in range(len(hole_c)):
                if index != len(hole_c) - 1:
                    edge = geom.LineString([hole_c[index], hole_c[index + 1]])
                    origins = edge.coords
                    while edge:
                        points, edge = self.cut(edge, rate)
                        sample[-1].append(points[0])
                        slope[-1].append(origins)
                else:
                    edge = geom.LineString([hole_c[index], hole_c[0]])
                    origins = edge.coords
                    while edge:
                        points, edge = self.cut(edge, rate)
                        sample[-1].append(points[0])
                        sample[-1].append(points[1])
                        slope[-1].append(origins)
                        slope[-1].append(origins)
        return sample, slope

    @staticmethod
    def cut(line, distance):
        # Cuts a line in two at a distance from its starting point
        if distance <= 0.0 or distance >= line.length:
            return list(line.coords), None

        return list(
            geom.LineString([line.coords[0], line.interpolate(distance)]).coords
        ), geom.LineString([line.interpolate(distance), line.coords[1]])

    def merge(self, polygons):
        """
        Merge a dataset of polygons, and separate them back if they are not touching each others and then forming a MultiPolygon
        :param polygons:
        :return:
        """
        new_polygons = []
        poly_union = unary_union(polygons)
        json_union = geom.mapping(poly_union)

        # Test if any MultiPolygons, and separate them
        if json_union["type"] == "MultiPolygon":
            for coords in json_union["coordinates"]:
                shell = coords[0]
                rings = []
                if len(coords) > 1:
                    rings = [coords[i] for i in range(1, len(coords))]
                new_feature = geom.Polygon(shell=shell, holes=rings)
                new_polygons.append(new_feature)
        else:
            new_polygons.append(poly_union)

        return new_polygons


def distance(datasetA, datasetB, match_matrix, svPts):
    r"""
    The distance function computes distances (in meters and in terms of angles too) between the two datasets A and B of polygons. Those polygons are sampled by multiple points on
    their edges. One dataset contains all the points sampling its polygons and the edges corresponding.

    For every polygon of dataset A which is matched with at least a polygon of B : every point of the A-polygon will be matched with the closest point on the sample of the matched B-polygon.
    From this match between samples-points will be computed an angle (edge to edge) and a distance, which are going to be returned in means.


    Args:
        datasetA (List): List composed of 2 elements : a list of its sampled polygons, a list of the edges of the polygons
        datasetB (List): List composed of 2 elements : a list of its sampled polygons, a list of the edges of the polygons
        match_matrix (Matrix): matrix of size len(A)xlen(B) representing the matches found between A and B
        svPts (boolean) : If True it will save the points whith the couple (distance, angle) of its match into a shapefile.
    Returns:
        mean_dist (List of float): List wich associates for A-polygon the mean minimum distance between each of its samples and a sample of a B-polygon matched with A. If no match for the A-polygon the value is set to -1.
        mean_angle (List): List wich associates for each A-polygon the mean angle between the edge of each of its samples and the edge of the closest B-polygon matched sample. If no match for the A-polygon the value is set to -1.
        index_A (int): A count of the number of polygons in the dataset A
        number_of_A_matched (int): A count of the number of matched polygons in the dataset A

    """
    sampledA, edgeA = datasetA[0], datasetA[1]
    sampledB, edgeB = datasetB[0], datasetB[1]

    index_A = 0
    mean_dist_by_A = defaultdict(float)
    mean_angle_by_A = defaultdict(float)
    max_dist_by_A = defaultdict(float)
    max_angle_by_A = defaultdict(float)
    nbr_of_A_matched = 0

    for match_instance in match_matrix:
        # match_matrix size AxB, match_instance is a binary list telling if A is matched with B[i]
        A_polygon, A_polygon_edge = (
            sampledA[index_A][0],
            edgeA[index_A][0],
        )  # Get the A-polygon sampled and its edges

        mean_dist_by_A[index_A] = -1  # For now, A is not matched
        nbr_matches = 0  # Number of matchs for A
        matrix_min = []
        matrix_edge = []

        for index_B in range(len(match_instance)):
            # for each polygon B test if there is a match
            if match_instance[index_B] == 1:  # there's a match between Bi and A !
                nbr_matches += 1
                B_polygon, B_polygon_edge = (
                    sampledB[index_B][0],
                    edgeB[index_B][0],
                )  # Get B-polygon sample & its edges

                if mean_dist_by_A[index_A] < 0:
                    # if it's the 1st match of A, initiate the dist to 0 and add the A-polygon as matched in the count.
                    mean_dist_by_A[index_A] = 0
                    nbr_of_A_matched += 1

                dist_to_this_match = (
                    []
                )  # list of the shortest distances from the sampled A to sampled B
                edge_match = []

                for icoord_A in range(len(A_polygon)):
                    # for each point sampled on the polygon A get its coordinates
                    coord_A = A_polygon[icoord_A]
                    point_A = geom.Point(coord_A)
                    sum_dist_point = []
                    for coord_B in B_polygon:
                        # for each point sampled on the polygon B get its coordinates
                        point_B = geom.Point(coord_B)
                        sum_dist_point.append(
                            point_A.distance(point_B)
                        )  # Shapely distance between points
                    dist_to_this_match.append(
                        min(sum_dist_point)
                    )  # Match the A-point to the closest B-point
                    edge_match.append(
                        B_polygon_edge[sum_dist_point.index(min(sum_dist_point))]
                    )

                matrix_min.append(
                    dist_to_this_match
                )  # matrix_min : nb of matches x nb of points sampled on A
                matrix_edge.append(edge_match)
            else:
                # If not matched: do nothing
                None

        if matrix_min:
            # if there is at least a match
            array_min = np.array(matrix_min).T
            array_edge = np.array(matrix_edge).transpose([1, 0, 2, 3])

            min_sum = 0
            max_dist = 0
            max_angle = 0
            angle_matching = []

            for index in range(len(array_min)):
                # for every point sampled on A-polygon :
                # get the A-edge on which the point is and the B-one the other point is
                # from that get the distance and the angle between the two of edges

                coord_A = A_polygon[index]
                dist = array_min[index].tolist()
                edges = array_edge[index].tolist()
                min_sum += min(dist)

                max_dist = min(dist) if min(dist) > max_dist else max_dist

                # Angle calculation
                edge_index = A_polygon_edge[index]
                edge_matched = edges[dist.index(min(dist))]

                x_vector_A = edge_index[1][0] - edge_index[0][0]
                y_vector_A = edge_index[1][1] - edge_index[0][1]

                x_vector_B = edge_matched[1][0] - edge_matched[0][0]
                y_vector_B = edge_matched[1][1] - edge_matched[0][1]

                angleA = math.atan2(y_vector_A, x_vector_A)
                angleB = math.atan2(y_vector_B, x_vector_B)

                if angleA > math.pi / 2:
                    angleA -= math.pi
                if angleA < -math.pi / 2:
                    angleA += math.pi

                if angleB > math.pi / 2:
                    angleB -= math.pi
                if angleB < -math.pi / 2:
                    angleB += math.pi

                angle = angleB - angleA

                deg_angle = abs(angle) * 180 / math.pi
                if deg_angle > 90:
                    deg_angle = 180 - deg_angle

                max_angle = deg_angle if deg_angle > max_angle else max_angle
                angle_matching.append(deg_angle)

                # if svPts:
                # TBD
                # shpOut = ""

                # schema = {
                #    'geometry': 'Point',
                #    'properties': {'id': int, 'distance': 'double', 'angle': 'float'},
                # }

                # with open(shpOut, "a", 'ESRI Shapefile', schema) as output:
                #    point = geom.Point(coord_A)

                # Write output
                #   output.write({'properties': {'id': len(output) + 1, 'distance': min(dist), 'angle': deg_angle},
                #                 'geometry': geom.mapping(point)
                #                 })

            if nbr_matches > 0:
                mean_dist_by_A[index_A] = min_sum / len(array_min)
                mean_angle_by_A[index_A] = sum(angle_matching) / len(angle_matching)
                max_dist_by_A[index_A] = max_dist
                max_angle_by_A[index_A] = max_angle
            else:  # if not matched
                mean_dist_by_A[index_A] = -1
                mean_angle_by_A[index_A] = -1
                max_dist_by_A[index_A] = -1
                max_angle_by_A[index_A] = -1

        index_A += 1  # next polygone A

    return (
        mean_dist_by_A,
        mean_angle_by_A,
        max_dist_by_A,
        max_angle_by_A,
        index_A,
        nbr_of_A_matched,
    )
