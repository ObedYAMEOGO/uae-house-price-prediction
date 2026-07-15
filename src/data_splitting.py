import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies.

    Splits a cleaned DataFrame into train/test sets, with the target column
    still present in both. Feature engineering (including any target
    transformation, e.g. log(Rent)) happens AFTER this split, fit only on the
    training set — separating features from target happens after that.
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The cleaned dataset (target column still included).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df)
        """
        raise NotImplementedError("Subclasses must implement the split_data method.")


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Basic random train-test split, with no stratification.

    Use when the dataset doesn't have a strong categorical imbalance that
    needs to be preserved across train/test (or as a quick baseline before
    trying a stratified approach).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(f"Performing simple train-test split ({self.test_size * 100:.0f}% test).")
        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        logging.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
        return train_df, test_df


class StratifiedSplitStrategy(DataSplittingStrategy):
    """
    Train-test split stratified by a categorical column, so that both splits
    preserve the same proportional representation of each category.

    Worth using here for `City`: it's heavily imbalanced (Dubai ~34k rows vs
    Umm Al Quwain ~65 rows), so a plain random split risks under-representing
    small cities in the test set. Stratifying keeps that proportion consistent.
    """

    def __init__(self, stratify_column: str, test_size: float = 0.2, random_state: int = 42):
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(
            f"Performing stratified train-test split on '{self.stratify_column}' "
            f"({self.test_size * 100:.0f}% test)."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.stratify_column],
        )
        logging.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
        return train_df, test_df


class DataSplitter:
    """Context class: delegates splitting to the selected strategy."""

    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy) -> None:
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df)


def split_features_and_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features (X) from the target (y). Call this AFTER feature
    engineering has been fit/applied — not before — since transformations
    like log(Rent) need the target column present during that step.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


if __name__ == "__main__":
    pass