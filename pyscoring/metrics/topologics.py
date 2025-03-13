from .matcher import MatchEngineIoMA
import numpy as np


class Topological_IoMA:
    r""" """

    def __init__(
        self,
        threshold=0.5,
        extend_matches=False,
        trim_invalid_geometry=False,
        autocorrect_invalid_geometry=False,
        return_match=True,
        am=False,
    ):

        self.trim_invalid_geometry = trim_invalid_geometry
        self.autocorrect_invalid_geometry = autocorrect_invalid_geometry

        self.threshold = threshold
        self.extend_matches = extend_matches
        self.return_match = return_match

        self.am = am

        self._init_values()

    def _init_values(self):
        # Matcher-graph
        self.list_matrices = []

        # Metrics
        self.numberP = 0
        self.numberGT = 0
        self.numberP_matched = 0
        self.numberGT_matched = 0

        self.GT_total_match = 0
        self.P_total_match = 0

        self._TFP = 0.0
        self._TFN = 0.0
        self._US = 0.0
        self._OS = 0.0

    def update(self, predictions, ground_truths):
        """
        Accumulates metrics for new detections and ground truths

        :param predictions: list of shapely.Polygon, the new detections to evaluate
        :param ground_truths: list of shapely.Polygon, the new ground truths to which we compare the detections
        """

        if len(predictions) == len(ground_truths) == 0:
            return
        elif len(ground_truths) == 0:
            self.numberP += len(predictions)
            return
        elif len(predictions) == 0:
            self.numberGT += len(ground_truths)
            return

        elif not self.am:
            match_engine = MatchEngineIoMA(
                self.threshold, extend_matches=self.extend_matches
            )
            match_matrix = match_engine.compute(predictions, ground_truths)
            self.list_matrices.append(match_matrix)
            return

        else:
            self.numberGT += len(ground_truths)
            self.numberGT_matched += len(ground_truths)
            self.numberP += len(predictions)
            self.numberP_matched += len(predictions)

    def compute(self):

        for matrix in self.list_matrices:
            self.numberP += matrix.shape[0]
            self.numberGT += matrix.shape[1]
            # Calculate for Predictions
            for row in matrix:
                total = np.sum(row)
                if total > 0:
                    self.numberP_matched += 1
                    self.P_total_match += total

            # Calculate for GT
            for column in matrix.T:
                total = np.sum(column)
                if total > 0:
                    self.numberGT_matched += 1
                    self.GT_total_match += total

        if self.numberP > 0:
            self._TFP = (self.numberP - self.numberP_matched) / self.numberP
            if self.numberP_matched > 0:
                self._US = self.P_total_match / self.numberP_matched

        if self.numberGT > 0:
            self._TFN = (self.numberGT - self.numberGT_matched) / self.numberGT
            if self.numberGT_matched > 0:
                self._OS = self.GT_total_match / self.numberGT_matched

        return (self._TFP, self._TFN, self._OS, self._US, self.numberP, self.numberGT)

    def reset(self):
        self._init_values()
