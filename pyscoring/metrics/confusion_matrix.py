# Third-party libraries
from sklearn.metrics import confusion_matrix

# Local libraries
from pyscoring.metrics.metric import Metric


class ConfusionMatrix(Metric):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix is such that is equal to the number of observations known to be in group
    and predicted to be in group.

    Args:
        labels (ndarray):  List of labels to index the matrix. This may be used to reorder or select a subset of
            labels. If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
        sample_weight (ndarray): Sample weights.
        normalize (‘true’, ‘pred’, ‘all’): Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be normalized.

    """

    def __init__(self, labels=None, sample_weight=None, normalize=None):

        # Init attributes
        self.detections = None
        self.ground_truths = None
        self.confusion_matrix = None

        # Set attrbiutes
        self.labels = labels
        self.sample_weight = sample_weight
        self.normalize = normalize

    def update(self, detections, ground_truths):
        """Accumulate detections and ground_truths.

        Args:
            detections (ndarray): A ndarray of detections.
            ground_truths (ndarray): A ndarray of ground truth.

        """
        detections = np.array(detections).flatten()
        ground_truths = np.array(ground_truths).flatten()
        if self.detections is None and self.ground_truths is None:
            self.detections = detections
            self.ground_truths = ground_truths
        else:
            if self.detections.shape != detections.shape:
                raise ValueError(
                    f"Detections must have same shapes. Got {self.detections.shape} and {detections.shape}."
                )
            if self.ground_truths.shape != ground_truths.shape:
                raise ValueError(
                    f"Ground truths must have same shapes. Got {self.ground_truths.shape} and {ground_truths.shape}."
                )

            # Accumulate
            self.detections += detections
            self.ground_truths += ground_truths

    def compute(self):
        """Compute confusion matrix.

        Returns:
            ndarray: Confusion matrix whose i-th row and j-th column entry indicates the number of samples
                     with true label being i-th class and prediced label being j-th class.

        """
        self.confusion_matrix = confusion_matrix(
            self.ground_truths,
            self.detections,
            labels=self.labels,
            sample_weight=self.sample_weight,
            normalize=self.normalize,
        )

        return self.confusion_matrix

    def reset(self):
        """This method automatically resets the metric state variables to their default value."""
        self.detections = None
        self.ground_truths = None
        self.confusion_matrix = None


if __name__ == "__main__":

    # Third-party libraries
    import numpy as np

    # Classification
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    metric = ConfusionMatrix()
    metric.update(y_pred, y_true)
    matrix = metric.compute()
    print(f"\nClassification:\n {matrix}")

    # Segmentation
    y_true = [[2, 0, 2], [2, 0, 1], [1, 1, 0]]
    y_pred = [[0, 0, 2], [2, 0, 2], [0, 0, 0]]
    metric = ConfusionMatrix()
    metric.update(y_pred, y_true)
    matrix = metric.compute()
    print(f"\nSegmentation:\n {matrix}")
