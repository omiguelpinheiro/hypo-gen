"""Classes used to apply transformations on data before use."""
import pandas as pd


class Standardizer:
    def __init__(self, data: pd.DataFrame):
        """Class used to standardize data

        Standardization means make the whole data have 0 mean and
        standard deviation equals to 1.

        Args:
            data (pd.DataFrame): The data from which to extract the
                mean and standard deviation.
        """
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize the data.

        Args:
            data (pd.DataFrame): The data to be standardized

        Returns:
            pd.DataFrame: The standardized data.
        """
        return (data - self.mean) / self.std

    def revert(self, data: pd.DataFrame) -> pd.DataFrame:
        """Revert the standardization done to data.

        Args:
            data (pd.DataFrame): Standardized data to be reverted.

        Returns:
            pd.DataFrame: The data after the process has been reverted.
        """
        return (data * self.std) + self.mean
