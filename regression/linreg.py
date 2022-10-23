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
        features_count: int,
        loss_func: MeanSquaredError = MeanSquaredError(),
        optimizer: Optimizer = GradientDescent(),
        seed: int | None = None,
    ):
        """
        Args:
            features_count (int): The amount of features in the model,
                this is used to create the weights of the linear
                regression model.
            loss_func (MeanSquaredError, optional): Which loss function
                to use to determine how wrong the model is. Defaults to
                MeanSquaredError().
            optimizer (Optimizer, optional): Which optimizer to use to
                make weights updates. Defaults to GradientDescent().
            seed (int | None, optional): A random number generator
                seed. Defaults to None.
        """
        super().__init__(features_count, loss_func, optimizer, seed)

    def solve(self, samples: np.ndarray, targets: np.ndarray):
        """Find the weights that gives the smallest error analitically.

        Args:
            samples (np.ndarray): Data to be used as training examples.
            targets (np.ndarray): Data to be used as training targets.
        """
        self.weights = (np.linalg.inv(samples.T @ samples) @ samples.T) @ targets

    def fit(
        self, samples: np.ndarray, targets: np.ndarray, epochs: int = 1000, quiet=True
    ):
        """Find the weights that gives the smallest error using the optimizer.

        Args:
            samples (np.ndarray): Data to be used as training examples.
            targets (np.ndarray): Data to be used as training targets.
            epochs (int, optional): The amount of weights updates
                iterations. Defaults to 1000.
            quiet (bool, optional): if it should give feedback about
                the training process. Defaults to True.
        """
        for epoch in range(epochs):
            predictions = self.predict(samples)
            grad = self.loss_func.grad(predictions, targets, samples)
            self.weights = self.optimizer.optimize(self.weights, grad)
            if not quiet and epoch % 100 == 0:
                print(self.loss(samples, targets))

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Make predictions for the samples.

        Args:
            samples (np.ndarray): The samples which we want to use
                to make the predictions.

        Returns:
            np.ndarray: The models predictions for the given samples.
        """
        return samples @ self.weights

    def loss(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """The model's loss for the samples.

        Args:
            samples (np.ndarray): The samples upon which the predictions
                will be made.
            targets (np.ndarray): The expected target of the samples.

        Returns:
            float: The model loss for the given samples.
        """
        predictions = self.predict(samples)
        return self.loss_func.eval(predictions, targets).sum() / len(targets)
