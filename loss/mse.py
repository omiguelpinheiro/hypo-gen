"""Definitions for using Mean Squared Error in machine learning."""
import numpy as np

from loss.loss import LossFunction


class MeanSquaredError(LossFunction):
    """Calculate the Mean Squared Error (MSE)."""

    def eval(self, predictions: np.ndarray, targets: np.ndarray):
        """Calculates loss based on predictions and targets.

        Args:
            predictions (np.ndarray): A models predictions.
            targets (np.ndarray): The expected value for the
                model predictions.
        """
        return ((predictions - targets) ** 2) / 2

    def grad(self, predictions: np.ndarray, targets: np.ndarray, samples: np.ndarray):
        """The gradient with respect to the model's parameters.

        Args:
            predictions (np.ndarray): A models predictions.
            targets (np.ndarray): The expected value for the
                model predictions.
            samples (np.ndarray): The original data that resulted
                in the predictions.
        """
        return ((predictions - targets) @ samples) / len(targets)
