"""Abstract class that models a loss function."""
from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def eval(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """Abstract class for a LossFunction.

        Calculates loss based on predictions and targets.

        Args:
            predictions (np.ndarray): A models predictions.
            targets (np.ndarray): The expected value for the
                model predictions.
        """

    @abstractmethod
    def grad(self, predictions: np.ndarray, targets: np.ndarray, samples: np.ndarray):
        """The gradient with respect to the model's parameters.

        Args:
            predictions (np.ndarray): A models predictions.
            targets (np.ndarray): The expected value for the
                model predictions.
            samples (np.ndarray): The original data that resulted
                in the predictions.
        """
