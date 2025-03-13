from math import floor, ceil

import numpy as np

import rasterio.features
from .confusion_matrix import ConfusionMatrix


class SurfacesMetric:
    r""" """

    def __init__(
        self,
        resolution=0.5,
        trim_invalid_geometry=False,
        autocorrect_invalid_geometry=False,
    ):

        self.trim_invalid_geometry = trim_invalid_geometry
        self.autocorrect_invalid_geometry = autocorrect_invalid_geometry
        self.path = "."
        self.resolution = resolution
        self._init_values()

    def _init_values(self):
        self._number_of_pix = 0

        self.confusion_matrix = np.array([[0, 0], [0, 0]])

        self._precision = 0.0
        self._rappel = 0.0

    def update(self, detections, ground_truths, boundingbox=None, resolution=None):
        """
        Accumulates metrics for new detections and ground truths

        :param detections: list of shapely.Polygon, the new detections to evaluate
        :param ground_truths: list of shapely.Polygon, the new ground truths to which we compare the detections
        :param boundingbox: list of coordinates defining the working zone to take into account in the metrics
        :param resolution: float, resolution to which we want to consider to surfacic evaluation of the data,
                            e.g. the resolution of the data source
        """
        if len(detections) == 0 or len(ground_truths) == 0:
            return

        resolution = self.resolution if resolution is None else resolution

        if boundingbox is None:
            # Select the union of the two BBOX
            bboxp = None
            bboxg = None

            for p in detections:
                if bboxp is None:
                    bboxp = list(p.bounds)
                else:
                    bboxp[0] = min(bboxp[0], list(p.bounds)[0])
                    bboxp[1] = min(bboxp[1], list(p.bounds)[1])
                    bboxp[2] = max(bboxp[2], list(p.bounds)[2])
                    bboxp[3] = max(bboxp[3], list(p.bounds)[3])
            bboxp = [
                bboxp[0] - resolution,
                bboxp[1] - resolution,
                bboxp[2] + resolution,
                bboxp[3] + resolution,
            ]

            for g in ground_truths:
                if bboxg is None:
                    bboxg = list(g.bounds)
                else:
                    bboxg[0] = min(bboxg[0], list(g.bounds)[0])
                    bboxg[1] = min(bboxg[1], list(g.bounds)[1])
                    bboxg[2] = max(bboxg[2], list(g.bounds)[2])
                    bboxg[3] = max(bboxg[3], list(g.bounds)[3])
            bboxg = [
                bboxg[0] - resolution,
                bboxg[1] - resolution,
                bboxg[2] + resolution,
                bboxg[3] + resolution,
            ]

            boundingbox = [0, 0, 0, 0]
            boundingbox[0] = min(bboxp[0], bboxg[0])
            boundingbox[1] = min(bboxp[1], bboxg[1])
            boundingbox[2] = max(bboxp[2], bboxg[2])
            boundingbox[3] = max(bboxp[3], bboxg[3])

        boundingbox = list(boundingbox)
        # if resolution provided then  make the bbox a multiple of the resolution
        if resolution > 0 and not None:
            boundingbox[0] = floor(boundingbox[0])
            boundingbox[1] = floor(boundingbox[1])
            boundingbox[2] = ceil(boundingbox[2] / resolution) * resolution
            boundingbox[3] = ceil(boundingbox[3] / resolution) * resolution

        width = int((boundingbox[2] - boundingbox[0]) / resolution)
        height = int((boundingbox[3] - boundingbox[1]) / resolution)

        output_shape = (height, width)
        trsf = rasterio.transform.from_bounds(*boundingbox, width, height)

        if len(ground_truths) > 0:
            mask_gt = rasterio.features.geometry_mask(
                ground_truths,
                output_shape,
                transform=trsf,
                invert=True,
                all_touched=True,
            )
        else:
            mask_gt = np.zeros(output_shape)

        if len(detections) > 0:
            mask_detect = rasterio.features.geometry_mask(
                detections, output_shape, transform=trsf, invert=True, all_touched=True
            )
        else:
            mask_detect = np.zeros(output_shape)

        self._number_of_pix += output_shape[0] * output_shape[1]

        confu_computer = ConfusionMatrix([0, 1])
        confu_computer.update(mask_detect, mask_gt)
        matrix = confu_computer.compute()
        self.confusion_matrix += matrix

    def compute(self):
        tp = self.confusion_matrix[1][1]  # Nb True Positives
        fp = self.confusion_matrix[0][1]  # Nb False Positives
        tn = self.confusion_matrix[0][0]  # Nb True Negatives
        fn = self.confusion_matrix[1][0]  # Nb False Negatives

        if tp == fp == 0:
            self._precision = 0  # PrSurf
            # print("Warning no pixel labelled in detection")
        else:
            self._precision = tp / (tp + fp)

        if tp == fn == 0:
            self._rappel = 0  # RapSurf
            # print("Warning no pixel labelled in GT")
        else:
            self._rappel = tp / (tp + fn)

        if self._number_of_pix != 0:
            tp = tp / self._number_of_pix
            fp = fp / self._number_of_pix
            tn = tn / self._number_of_pix
            fn = fn / self._number_of_pix
        else:
            tp = fp = tn = fn = 0

        return (tp, tn, fp, fn, self._precision, self._rappel, self._number_of_pix)

    def reset(self):
        self._init_values()
