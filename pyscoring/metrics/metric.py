# Standard libraries
from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self) -> None:
        """Override this method to update the state variables of your metric class."""
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """Override this method to compute the final metric value from state variables."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """This method automatically resets the metric state variables to their default value."""
        raise NotImplementedError
