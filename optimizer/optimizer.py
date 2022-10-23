"""Abstract class that models a model optimizer."""
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Abstract class for a model optimizer."""

    @abstractmethod
    def optimize(
        self, weights: np.ndarray, grad: np.ndarray, learning_rate: float = 0.001
    ) -> np.ndarray:
        """Calculates the weights update.

        Returns:
            np.ndarray: The model weights updated once.
        """
