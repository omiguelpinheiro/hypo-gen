"""Definitions for using Linear Regression algorithm."""

import numpy as np

from loss.mse import MeanSquaredError
from mtype.weighted import WeightedModel
from optimizer.optimizer import Optimizer
from optimizer.gradient import GradientDescent


class LinearRegression(WeightedModel):
    """A LinearRegression model.

    This model tries to find a linear relationship between it's
    features and some target."""

    def __init__(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        loss_func: MeanSquaredError = MeanSquaredError(),
        optimizer: Optimizer = GradientDescent(),
        seed: int | None = None,
    ):
        """
        Args:
            samples (np.ndarray): The data that will be used to train the model.
            targets (np.ndarray): The expected targets for the samples.
            loss_func (MeanSquaredError, optional): Which loss function
                to use to determine how wrong the model is. Defaults to
                MeanSquaredError().
            optimizer (Optimizer, optional): Which optimizer to use to
                make weights updates. Defaults to GradientDescent().
            seed (int | None, optional): A random number generator
                seed. Defaults to None.
        """
        super().__init__(samples, targets, loss_func, optimizer, seed)

    def solve(self):
        """Find the weights that gives the smallest error analitically."""
        self.weights = (
            np.linalg.inv(self.samples.T @ self.samples) @ self.samples.T
        ) @ self.targets

    def fit(self, epochs: int = 1000, quiet=True):
        """Find the weights that gives the smallest error using the optimizer.

        Args:
            epochs (int, optional): The amount of weights updates
                iterations. Defaults to 1000.
            quiet (bool, optional): if it should give feedback about
                the training process. Defaults to True.
        """
        for epoch in range(epochs):
            predictions = self.predict(self.samples, has_dummy=True)
            grad = self.loss_func.grad(predictions, self.targets, self.samples)
            self.weights = self.optimizer.optimize(self.weights, grad)
            if not quiet and epoch % 100 == 0:
                print(self.loss(self.samples, self.targets, has_dummy=True))

    def predict(self, samples: np.ndarray, has_dummy=False) -> np.ndarray:
        """Make predictions for the samples.

        Args:
            samples (np.ndarray): The samples which we want to use
                to make the predictions.
            has_dummy (bool): If the provided data already has
                the dummy feature.

        Returns:
            np.ndarray: The models predictions for the given samples.
        """
        if not has_dummy:
            samples = self._add_dummy(samples)
        return samples @ self.weights

    def loss(self, samples: np.ndarray, targets: np.ndarray, has_dummy=False) -> float:
        """The model's loss for the samples.

        Args:
            samples (np.ndarray): The samples upon which the predictions
                will be made.
            targets (np.ndarray): The expected target of the samples.
            has_dummy (bool): If the provided data already has
                the dummy feature.

        Returns:
            float: The model loss for the given samples.
        """
        predictions = self.predict(samples, has_dummy=has_dummy)
        return self.loss_func.eval(predictions, targets).sum() / len(targets)
