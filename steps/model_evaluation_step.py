import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from zenml import client, step

import src.model_evaluation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

experiment_tracker = client.Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    trained_model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[dict, float]:
    """
    Evaluates the trained model on the held-out test set and logs metrics
    into the SAME MLflow run that model_building_step logged the model to
    (both steps declare experiment_tracker, so ZenML keeps them linked).
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    if isinstance(trained_model, Pipeline):
        raise TypeError(
            "trained_model must be a plain fitted estimator, not a Pipeline — "
            "feature engineering happens separately, upstream, in this project."
        )

    logging.info("Starting model evaluation...")

    evaluator = src.model_evaluation.ModelEvaluator(strategy=src.model_evaluation.RegressionModelEvaluationStrategy())
    evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    mae_aed = evaluation_metrics.get("MAE (AED)")
    logging.info(f"Model evaluation completed. MAE (AED): {mae_aed:.2f}")

    for metric_name, value in evaluation_metrics.items():
        safe_name = metric_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        mlflow.log_metric(safe_name, value)

    return evaluation_metrics, mae_aed