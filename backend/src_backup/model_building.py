import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
import yaml
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelBuildingStrategy(ABC):
    """
    Abstract base class for model-building strategies.

    Enforces a common build_and_train_model method so ModelBuilder can
    swap between model types without changing any calling code.
    """

    @abstractmethod
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RegressorMixin:
        raise NotImplementedError("Subclasses must implement build_and_train_model.")


class LinearRegressionStrategy(ModelBuildingStrategy):
    """
    Linear Regression with standard scaling.

    Serves as the baseline model — useful mainly as a reference point to
    confirm the pipeline works end-to-end and to measure how much a
    tree-based model actually improves over a simple linear fit.
    """

    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")
        return pipeline


class RandomForestStrategy(ModelBuildingStrategy):
    """
    Random Forest Regressor — no scaling needed (tree splits are scale-invariant).

    Well suited to this dataset: handles non-linear interactions between
    Area_in_sqft, Beds, City, Type, and the many one-hot encoded Location
    categories better than a linear model, and is robust to the remaining
    mild skew even after log-transforming the target.
    """

    def __init__(self, **hyperparameters: Any):
        self.hyperparameters = hyperparameters

    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestRegressor:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info(f"Initializing Random Forest with hyperparameters: {self.hyperparameters}")
        model = RandomForestRegressor(**self.hyperparameters)

        logging.info("Training Random Forest model.")
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model


class ModelBuilder:
    """Context class: delegates model building/training to the selected strategy."""

    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy) -> None:
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


class ModelBuilderFactory:
    """
    Factory that reads configs/model_config.yaml and returns the correctly
    configured ModelBuildingStrategy — keeps model selection and
    hyperparameters out of code, so switching models is a config edit,
    not a code change.
    """

    @staticmethod
    def from_config(config_path: str = "configs/model_config.yaml") -> ModelBuildingStrategy:
        with open(config_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        model_name = config["model"]["name"]

        if model_name == "linear_regression":
            return LinearRegressionStrategy()

        if model_name == "random_forest":
            hyperparameters = config.get("random_forest", {})
            return RandomForestStrategy(**hyperparameters)

        raise ValueError(f"Unsupported model name in config: '{model_name}'")


if __name__ == "__main__":
    pass