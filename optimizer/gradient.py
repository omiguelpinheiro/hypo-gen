"""Definitions for using Gradient Descent algorithms to optimize
a model's weights."""

import numpy as np

from optimizer.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        """Calculate weights updates using the Gradient Descent method.

        TODO: Make this class usable for Stochastic Gradient Descent (SGD)
        Mini Batch Gradient Descent (MBGD), and Batch Gradient Descent (BGD)
        based on the batch size. batch_size = 1 means SGD,
        1 < batch_size < len(data) means MBGD and batch_size = len(data) means
        BGD.

        Args:
            learning_rate (float, optional): The size of the
                Grandient Descent step. Defaults to 0.001.
        """
        self.learning_rate = learning_rate

    def optimize(
        self, weights: np.ndarray, grad: np.ndarray, learning_rate: float = 0.001
    ) -> np.ndarray:
        """Calculates the weights update using the Gradient Descent algorithm.

        Args:
            weights (np.ndarray): The original weights.
            grad (np.ndarray): The gradient of loss function with respect
                to the original weights.
            learning_rate (float, optional): The size of the step in each
                weight update. Defaults to 0.001.

        Returns:
            np.ndarray: The new weights after the weights were
                updated once.
        """
        return weights - learning_rate * grad
