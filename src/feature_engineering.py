import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineeringStrategy(ABC):
    """
    Blueprint for feature transformation strategies.

    fit() learns any parameters (means, categories, thresholds) from training
    data ONLY. transform() applies those learned parameters to any dataset
    (train, test, or a single new row at inference time). This split prevents
    data leakage and makes the fitted state persistable for serving.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureEngineeringStrategy":
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method — use ONLY on training data."""
        return self.fit(df).transform(df)


class RareCategoryBucketingStrategy(FeatureEngineeringStrategy):
    """
    Groups infrequent categories in a categorical column into a single 'Other'
    bucket, based on frequency learned from training data.

    Use before one-hot encoding a high-cardinality column (e.g. this dataset's
    Location column: 441 unique values, but only 120 have >= 100 listings).
    """

    def __init__(self, feature: str, min_count: int = 100, other_label: str = "Other"):
        self.feature = feature
        self.min_count = min_count
        self.other_label = other_label
        self.frequent_categories_: Optional[pd.Index] = None

    def fit(self, df: pd.DataFrame) -> "RareCategoryBucketingStrategy":
        counts = df[self.feature].value_counts()
        self.frequent_categories_ = counts[counts >= self.min_count].index
        logging.info(
            f"'{self.feature}': keeping {len(self.frequent_categories_)} categories "
            f"with >= {self.min_count} occurrences; rest bucketed as '{self.other_label}'."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.frequent_categories_ is None:
            raise RuntimeError("Call fit() before transform().")

        df_transformed = df.copy()
        df_transformed[self.feature] = df_transformed[self.feature].where(
            df_transformed[self.feature].isin(self.frequent_categories_), self.other_label
        )
        return df_transformed


class LogTransformation(FeatureEngineeringStrategy):
    """
    Log(1+x) transform for skewed numerical features. Skips any feature not present in the given DataFrame, since at
    inference time the target column (Rent) won't exist in the input yet.
    """

    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, df: pd.DataFrame) -> "LogTransformation":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        for feature in self.features:
            if feature in df_transformed.columns:
                df_transformed[feature] = np.log1p(df_transformed[feature])
            else:
                logging.warning(
                    f"Feature '{feature}' not found, skipping log transform "
                    f"(expected at inference time, since the target isn't known yet)."
                )
        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):
    """Z-score normalization (mean=0, std=1). Fit ONLY on training data."""

    def __init__(self, features: List[str]):
        self.features = features
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "StandardScaling":
        self.scaler.fit(df[self.features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(df[self.features])
        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    """Scales features to a fixed range (default 0-1). Fit ONLY on training data."""

    def __init__(self, features: List[str], feature_range: tuple = (0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, df: pd.DataFrame) -> "MinMaxScaling":
        self.scaler.fit(df[self.features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(df[self.features])
        return df_transformed


class OneHotEncoding(FeatureEngineeringStrategy):
    """
    One-hot encodes categorical features. Fit ONLY on training data so the
    category list is fixed; unseen categories at inference are safely ignored
    (handle_unknown="ignore") rather than crashing the API.
    """

    def __init__(self, features: List[str]):
        self.features = features
        self.encoder = OneHotEncoder(
            sparse_output=False, drop="first", handle_unknown="ignore"
        )

    def fit(self, df: pd.DataFrame) -> "OneHotEncoding":
        self.encoder.fit(df[self.features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()

        encoded_array = self.encoder.transform(df[self.features])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(self.features),
            index=df_transformed.index,
        )

        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed


class FeatureEngineer:
    """
    Runs a sequence of FeatureEngineeringStrategy steps, in order, and allows
    persisting the fitted pipeline (via joblib) for reuse at inference time.
    """

    def __init__(self):
        self._steps: List[FeatureEngineeringStrategy] = []

    def add_step(self, strategy: FeatureEngineeringStrategy) -> "FeatureEngineer":
        self._steps.append(strategy)
        return self

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fits every step in sequence. Call ONLY on training data."""
        current_df = df
        for step in self._steps:
            step.fit(current_df)
            current_df = step.transform(current_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies every already-fitted step in sequence. Safe on train, test, or new data."""
        current_df = df
        for step in self._steps:
            current_df = step.transform(current_df)
        return current_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method — use ONLY on training data."""
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        """Persists the fitted pipeline."""
        joblib.dump(self, path)
        logging.info(f"Feature engineering pipeline saved to {path}")

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        """Loads a previously fitted pipeline"""
        return joblib.load(path)


if __name__ == "__main__":
    pass