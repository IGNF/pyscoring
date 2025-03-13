import numpy as np
import rasterio
from rasterio import features
import rtree
import shapely
from shapely.errors import TopologicalError


class MatchEngine:
    """
    The MatchEngine allows to match a polygon with other polygons according to its criterion
    For now only the IoMA (Intersection Over Minimum Area) and the IoU (Intersection Over Union) criteria are developed

    Each new criterion inherits from MatchEngine the functions compute and matcher function but changing the condition
    of the match
    """

    def __init__(self, match_algorithm, extend_matches=False):
        """
        :param match_algorithm: String, defining the criterion used by our matcher
        :param extend_matches: Boolean, if False the polygons will be compared using their own borders,
                                        if True the polygons will be compared with a fusion of polygons touching them
        """
        assert match_algorithm in ["ioma", "iou"], "matcher only works with ioma or iou"

        self.match_algorithm = match_algorithm
        self.extend_matches = extend_matches

    def compute(self, predictions, ground_truths):
        """
        Launching the match between the list of polygons of the prediction and the one of the ground_truth

        :param predictions: list of shapely.Polygon, list of the polygons of the prediction
        :param ground_truths: list of shapely.Polygon, list of the polygons of the ground_truth
        :return: match_matrix : np.array(len(predictions),len(ground_truths)), matrix matching the polygons w bool
        """
        match_matrix = np.zeros((len(predictions), len(ground_truths)))
        spatial_index = rtree.index.Index()

        # Process the GT
        i_gt = 0
        for gt in ground_truths:
            bbox = rasterio.features.bounds(gt)
            spatial_index.insert(id=i_gt, coordinates=bbox)
            i_gt += 1

        # Process the predictions
        i_p = 0
        for p in predictions:
            match_matrix = self.matcher(
                p, i_p, ground_truths, spatial_index, match_matrix
            )
            i_p += 1

        return match_matrix

    def matcher(self, polygon, id, list_gt, spatial_index, match_matrix):
        """
        Find the polygons of the GT matching with the given polygon, updating the match_matrix when found

        :param polygon: shapely.Polygon, a Polygon of the prediction
        :param id: int, the position of the polygon in the predictions' list
        :param list_gt: list of shapely.Polygon, list of the polygons of the ground_truth
        :param spatial_index: rtree.index.Index, spatial index of the GT polygons
        :param match_matrix: np.array, the match matrix to update with the found matchs
        :return: out_matrix: np.array, the match_matrix updated
        """
        out_matrix = match_matrix
        bbox = rasterio.features.bounds(polygon)
        matched = False

        # for each gt_bbox (in the index) intersecting the p_bbox, calculate IoMA
        list_possible_matches = [
            list_gt[number_index] for number_index in spatial_index.intersection(bbox)
        ]
        list_indexes = [
            number_index for number_index in spatial_index.intersection(bbox)
        ]

        for i in range(len(list_possible_matches)):
            number_index = list_indexes[i]
            possible_match = list_possible_matches[i]
            try:
                if self.condition(polygon, possible_match):
                    out_matrix[id][number_index] = 1
                    matched = True
            except TopologicalError:
                pass

        if self.extend_matches and not matched and len(list_possible_matches) > 1:

            union_matches = shapely.unary_union(list_possible_matches)
            try:
                if self.condition(polygon, union_matches):
                    for number_index in list_indexes:
                        out_matrix[id][number_index] = 1
            except TopologicalError:
                pass
        return out_matrix

    def condition(self, polygon, possible_match):
        """

        :param polygon: shapely.Polygon, the polygon to match
        :param possible_match: shapely.Polygon, a polygon crossing its bbox is a possible match
        :return: boolean, True as no condition is defined so we match every polygon crossing the bbox
        """
        print("No condition defined, will match every polygons with crossing bboxs")
        return True


class MatchEngineIoMA(MatchEngine):
    """
    Intersection Over Minimum Area :
    According to this criterion two polygons are considered to match if the area of their intersection divided by
    the area of the smallest one is greater than the chosen threshold
    """

    def __init__(self, threshold=0.5, strict=False, extend_matches=False):
        self.threshold = threshold
        self.strict = strict
        super().__init__("ioma", extend_matches)

    def condition(self, polygon, possible_match):
        """
        Condition IoMA, computing the IoMA of both the polygons and comparing it to the chosen threshold

        :param polygon: shapely.Polygon, the polygon to match
        :param possible_match: shapely.Polygon, a polygon crossing its bbox is a possible match
        :return: boolean, True if ioma>threshold there is a match, if not False
        """
        try:
            I = shapely.intersection(polygon, possible_match)
            ioma = I.area / float(min(polygon.area, possible_match.area))

            if self.strict:
                if ioma > self.threshold:
                    return True
            else:
                if ioma >= self.threshold:
                    return True

        except TopologicalError:
            return False
        return False


class MatchEngineIoU(MatchEngine):
    """
    Intersection Over Union :
    According to this criterion two polygons are considered to match if the area of their intersection divided by
    the area of their union is greater than the chosen threshold
    """

    def __init__(self, threshold=0.7, strict=False, extend_matches=False):
        self.threshold = threshold
        self.strict = strict
        super().__init__("iou", extend_matches)

    def condition(self, polygon, possible_match):
        """
        Condition IoU, computing the IoU of both the polygons and comparing it to the chosen threshold

        :param polygon: shapely.Polygon, the polygon to match
        :param possible_match: shapely.Polygon, a polygon crossing its bbox is a possible match
        :return: boolean, True if iou>threshold there is a match, if not False
        """
        try:
            I = shapely.intersection(polygon, possible_match)
            U = shapely.union(polygon, possible_match)
            iou = I.area / U.area

            if self.strict:
                if iou > self.threshold:
                    return True
            else:
                if iou >= self.threshold:
                    return True

        except TopologicalError:
            return False
