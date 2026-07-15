import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import ArtifactConfig, Model, client, step

from src.model_building import ModelBuilderFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

experiment_tracker = client.Client().active_stack.experiment_tracker

model = Model(
    name="uae_house_price_predictor",
    version=None,
    license="Apache 2.0",
    description="UAE property rent prediction model.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_path: str = "configs/model_config.yaml",
) -> Annotated[
    RegressorMixin, ArtifactConfig(name="trained_model", is_model_artifact=True)
]:
    """
    Trains a regression model on already-processed features.

    No manual mlflow.start_run()/end_run() here — ZenML opens and manages
    the MLflow run automatically because experiment_tracker is attached to
    this step's decorator. This keeps the run open across this step AND
    model_evaluation_step (which also declares experiment_tracker), so both
    the model and its evaluation metrics land in the SAME MLflow run instead
    of two disconnected ones.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    strategy = ModelBuilderFactory.from_config(config_path)
    logging.info(f"Selected model strategy: {strategy.__class__.__name__}")

    mlflow.sklearn.autolog()

    logging.info(f"Training model on {X_train.shape[0]} rows, {X_train.shape[1]} features.")
    trained_model = strategy.build_and_train_model(X_train, y_train)
    logging.info("Model training completed successfully.")

    return trained_model