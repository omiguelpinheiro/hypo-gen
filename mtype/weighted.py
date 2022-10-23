"""Definitions for a model that uses linear weights."""

import numpy as np
from loss.loss import LossFunction

from optimizer.optimizer import Optimizer


class WeightedModel:
    def __init__(
        self,
        features_count: int,
        loss_func: LossFunction,
        optimizer: Optimizer,
        seed: int | None = None,
    ):
        """A model that learns based on linear weights.

        Args:
            features_count (int): The amount of features in the model.
            loss_func (MeanSquaredError, optional): Which loss function
                to use to determine how wrong the model is.
            optimizer (Optimizer, optional): Which optimizer to use to
                make weights updates.
            seed (int | None, optional): A random number generator
                seed. Defaults to None.
        """
        self.features_count = features_count
        self.loss_func = loss_func
        self.optimizer = optimizer

        rand_gen = np.random.RandomState(seed)
        self.weights = rand_gen.rand(features_count)
