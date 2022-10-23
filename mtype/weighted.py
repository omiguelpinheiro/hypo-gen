"""Definitions for a model that uses linear weights."""

import numpy as np
from loss.loss import LossFunction

from optimizer.optimizer import Optimizer


class WeightedModel:
    def __init__(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        loss_func: LossFunction,
        optimizer: Optimizer,
        seed: int | None = None,
    ):
        """A model that learns based on linear weights.

        Args:
            samples (np.ndarray): The data that will be used to train the model.
            targets (np.ndarray): The expected targets for the samples.
            loss_func (MeanSquaredError, optional): Which loss function
                to use to determine how wrong the model is.
            optimizer (Optimizer, optional): Which optimizer to use to
                make weights updates.
            seed (int | None, optional): A random number generator
                seed. Defaults to None.
        """
        self.samples = self._add_dummy(samples)
        self.targets = targets
        self.loss_func = loss_func
        self.optimizer = optimizer

        rand_gen = np.random.RandomState(seed)
        self.weights = rand_gen.rand(self.samples.shape[1])

    def _add_dummy(self, samples: np.ndarray) -> np.ndarray:
        """Add the dummy feature to the samples.

        Args:
            samples (np.ndarray): Samples without the dummy feature.

        Returns:
            np.ndarray: Samples with dummy features.
        """
        dummy_feature = np.ones(len(samples))
        return np.c_[samples, dummy_feature]
