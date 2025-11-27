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
        * :meth:`compute` which computes PCont, RCont per label from accumulated values
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
        self._number_of_samples_GT = 0 
        self._number_of_samples_P = 0

        self._ground_truth_labels = set()

        self.summ_distPrecision = 0.0
        self.summ_distRecall = 0.0
        self.maxDistP = 0.0
        self.maxDistR = 0.0

        self._numberDmatched = 0
        self._numberGTmatched = 0

        self._P = 0.0  # Precision metric
        self._R = 0.0  # Recall metric

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
        else : 
            print('Not matched data. Pass')
            return 
            
        # Sampling the detections
        sampled_detection = []
      
        total_notccw = 0

        for polygon in polygon_detect:
            linRing = geom.LinearRing(polygon.exterior)

            if not linRing.is_ccw:  # test the outside line
                total_notccw += 1

            sampled_polygon = self._sampling(polygon, self.pixel_size)

            sampled_detection.append(sampled_polygon)

        # Sampling the gt
        sampled_gt = []
        total_notccwgt = 0
        for polygon in polygon_gt:
            linRing = geom.LinearRing(polygon.exterior)
            if not linRing.is_ccw:
                total_notccwgt += 1

            sampled_polygon = self._sampling(polygon, self.pixel_size)

            sampled_gt.append(sampled_polygon)

        # for each polygon in the dataset of detections : compute the means with the GT
        if mode == 0 or mode == 1:
            (
                mean_dist_by_polygon,
                max_dist_by_polygon,
                nb_pts_by_polygon,
                nbDmatched,
            ) = distance(
                sampled_detection,
                sampled_gt,
                match_matrix,
                self.svPts,
            )

            mean_dist = 0
            max_dist = 0
            nb_pts = 0

            for i, mean_i in enumerate(mean_dist_by_polygon):
                if mean_i != -1 :
                    mean_dist += mean_i
                    max_dist = max_dist_by_polygon[i] if max_dist_by_polygon[i] > max_dist else max_dist
                    nb_pts += nb_pts_by_polygon[i]
            
            self.summ_distPrecision += mean_dist * nb_pts
            self.maxDistP = max_dist if max_dist > self.maxDistP else self.maxDistP
            self._number_of_samples_P += nb_pts
            self._numberDmatched += nbDmatched

        # for each polygon in the dataset of GT : compute the means with the predictions
        if mode == 0 or mode == 2:
            (
                mean_dist_by_polygon,
                max_dist_by_polygon,
                nb_pts_by_polygon,
                nbGTmatched,
            ) = distance(
                sampled_gt,
                sampled_detection,
                match_matrix.T,
                self.svPts,
            )

            mean_dist = 0
            max_dist = 0
            nb_pts = 0

            for i, mean_i in enumerate(mean_dist_by_polygon):
                if mean_i != -1 :
                    mean_dist += mean_i
                    max_dist = max_dist_by_polygon[i] if max_dist_by_polygon[i] > max_dist else max_dist
                    nb_pts += nb_pts_by_polygon[i]
            
            self.summ_distRecall = mean_dist * nb_pts
            self.maxDistR = max_dist if max_dist > self.maxDistR else self.maxDistR
            self._number_of_samples_GT += nb_pts
            self._numberGTmatched += nbGTmatched

        self._number_of_dataset += 1

    def compute(self):
        if self._number_of_samples_P and self._number_of_samples_GT:
            self._P = self.summ_distPrecision / self._number_of_samples_P
            self._R = self.summ_distRecall / self._number_of_samples_GT

        return (
            self._P,
            self._R,
            self.maxDistP,
            self.maxDistR,
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

        exterior_c = polygon.exterior.coords
        

        
        for i, coord in enumerate(exterior_c):            
            previous_coord = exterior_c[i-1]

            if i==0 and coord == previous_coord :
                pass            
            else : 
                previous_coord = exterior_c[i-1]
                
                edge = geom.LineString([geom.Point(previous_coord), geom.Point(coord)])
                pts_edge = self.interpolate(edge, rate)[:-1]
                sample[0] += pts_edge        

        for h_id, hole in enumerate(polygon.interiors):
            hole_c= hole.coords
            sample.append([])
           
            for i, coord in enumerate(hole_c):
                previous_coord = exterior_c[i-1]

                if i==0 and coord == previous_coord :
                    pass            
                else : 
                    previous_coord = exterior_c[i-1]
                    edge = geom.LineString([geom.Point(previous_coord), geom.Point(coord)])
                    pts_edge = self.interpolate(edge, rate)[:-1]
                    sample[0] += pts_edge 
                    #sample[h_id+1] += pts_edge 
            
        return sample

    @staticmethod
    def floatrange(start, stop, step):
        list = []
        while start <= stop :
            list.append(start)
            start += step 
        
        return list

    def interpolate(self, line, rate):
        pts = []
        #TODO : 
        #class_id = row_gdf.get(CLASS_ROW, 99) if isgt else row_gdf.get(CLASS_ROW, 0)
        #class_id = CLASS_NAMES[int(class_id)]
        #
        try:
            for step in self.floatrange(0, line.length, rate):
                pts.append(line.line_interpolate_point(step))
        except Exception as e:
            print("Error w :", line)
            print(e)
    
        return pts #, class_id
    
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


def distance(sampledA, sampledB, match_matrix, svPts=False):
    r"""
    The distance function computes distances (in meters and in terms of angles too) between the two datasets A and B of polygons. Those polygons are sampled by multiple points on
    their edges. One dataset contains all the points sampling its polygons and the edges corresponding. For every polygon of dataset A which is matched with at least a polygon of B : every point of the A-polygon will be matched with the closest point on the sample of the matched B-polygon.
    Deprecated : From this match between samples-points will be computed an angle (edge to edge) and a distance, which are going to be returned in means.


    Args:
        datasetA (List): List composed of 2 elements : a list of its sampled polygons, a list of the edges of the polygons
        datasetB (List): List composed of 2 elements : a list of its sampled polygons, a list of the edges of the polygons
        match_matrix (Matrix): matrix of size len(A)xlen(B) representing the matches found between A and B
        svPts (boolean) : If True it will save the points whith the distance of its match into a shapefile
    Returns:
        mean_dist (List of float): List wich associates for A-polygon the mean minimum distance between each of its samples and a sample of a B-polygon matched with A. If no match for the A-polygon the value is set to -1
        index_A (int): A count of the number of polygons in the dataset A
        number_of_A_matched (int): A count of the number of matched polygons in the dataset A

    """

    
    mean_dist_by_A = [-1] * len(sampledA)
    max_dist_by_A = [-1] * len(sampledA)
    nb_samples_by_A = [-1] * len(sampledA)

    nbr_of_A_matched = 0

    for ia, match_instance in enumerate(match_matrix):
        # match_matrix size AxB, match_instance is a binary list telling if A is matched with B[i]
        sp_A = sampledA[ia][0] # sampled polygon A 

        nbr_matches = 0  # Number of matchs for A
        list_min = [-1] * len(sp_A) # initialise a list of distance for each point sampled on A 

        for ib, match_bool in enumerate(match_instance):
            # for each polygon B test if there is a match
            if match_bool == 1:  # there's a match between Bi and A !
                nbr_matches += 1
                nbr_of_A_matched += 1

                sp_B = sampledB[ib][0] # sampled polygon B

                for ic, point_A in enumerate(sp_A):
                    # for each point sampled on the polygon A get its coordinates
                    minimum_dist = None
                    list_dist_point = point_A.distance(sp_B)
                                           
                    minimum_dist = np.min(list_dist_point) # Match the A-point to the closest B-point
                    
                    if nbr_matches == 1 : # first match of A with B so initialisation of min distance
                        list_min[ic] = minimum_dist
                    else : # not the first match so lets check if we encountered a closer point
                        list_min[ic] = minimum_dist if minimum_dist < list_min[ic] else list_min[ic]
                    
                    if svPts:
                        shpOut = "/home/ELe-Bihan/gitclones/github.com/pyscoring/example/2D/output/pts.shp"
                        schema = {
                            'geometry': 'Point',
                            'properties': {'id': int, 'distance': 'double', 'angle': 'float'},
                        }
                        with open(shpOut, "a", 'ESRI Shapefile', schema) as output:
                            point = point_A
                        
                        output.write({'properties': {'id': len(output) + 1, 'distance': min(list_dist_point)},
                                        'geometry': geom.mapping(point)
                                        })

            else : # If not matched: do nothing
                pass        

        if nbr_matches > 0:
            mean_dist_by_A[ia] = np.mean(list_min)           
            max_dist_by_A[ia] = np.max(list_min)
            nb_samples_by_A[ia] = len(list_min)

            if np.mean(list_min) > 5 and np.mean(list_min) < 6 :
                print(list_min)
                print(sp_A)

        else:  # if not matched
           pass
    return (
        mean_dist_by_A,
        max_dist_by_A,
        nb_samples_by_A,
        nbr_of_A_matched,
    )