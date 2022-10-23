"""Utilities to manipulate data."""
import pandas as pd


def train_test_split(
    data: pd.DataFrame,
    ratio: float = 0.75,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train and test datasets.

    Args:
        data (pd.DataFrame): Data to be splited.
        ratio (float, optional): A number from 0 to 1 meaning
            a percentage of the data that should be used as
            training data, the rest will be used as test
            data. Defaults to 0.75.
        seed (int | None, optional): A random number generator
            seed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Respectively
            the train samples, train targets, test samples
            and sample targets.
    """
    # TODO: Create argument to specify which column is the target column
    data = data.sample(frac=1, ignore_index=True, random_state=seed)

    train = data[: int(len(data) * ratio)]
    samples_train = train.iloc[:, :-1]
    targets_train = train.iloc[:, -1]

    test = data[int(len(data) * ratio) :]
    samples_test = test.iloc[:, :-1]
    targets_test = test.iloc[:, -1]

    return samples_train, targets_train, samples_test, targets_test
