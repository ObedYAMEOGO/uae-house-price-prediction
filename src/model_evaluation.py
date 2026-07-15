import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelEvaluationStrategy(ABC):
    """Abstract base class for model evaluation strategies."""

    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement evaluate_model.")


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    """
    Evaluates a regression model on both the log-transformed target (the
    space the model was actually trained on) AND the real AED scale (after
    reversing log1p), since Rent was log-transformed during feature
    engineering to handle its severe skew (83.7).

    Log-space metrics are the fair, stable comparison point between model
    runs. Real-AED-space metrics (especially MAE) are what you'd actually
    put in a portfolio README, since "on average, off by X AED" is far more
    meaningful to a reader than a log-scale error number.
    """

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series.")

        logging.info("Generating predictions using the trained model.")
        y_pred_log = model.predict(X_test)

        # --- Log-space metrics (the space the model was trained on) ---
        mse_log = mean_squared_error(y_test, y_pred_log)
        rmse_log = np.sqrt(mse_log)
        mae_log = mean_absolute_error(y_test, y_pred_log)
        r2_log = r2_score(y_test, y_pred_log)

        # --- Real AED-space metrics (reverse the log1p transform) ---
        y_test_aed = np.expm1(y_test)
        y_pred_aed = np.expm1(y_pred_log)

        mae_aed = mean_absolute_error(y_test_aed, y_pred_aed)
        rmse_aed = np.sqrt(mean_squared_error(y_test_aed, y_pred_aed))
        r2_aed = r2_score(y_test_aed, y_pred_aed)

        metrics = {
            "MSE (log-space)": mse_log,
            "RMSE (log-space)": rmse_log,
            "MAE (log-space)": mae_log,
            "R-Squared (log-space)": r2_log,
            "MAE (AED)": mae_aed,
            "RMSE (AED)": rmse_aed,
            "R-Squared (AED)": r2_aed,
        }

        logging.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics


class ModelEvaluator:
    """Context class: delegates evaluation to the selected strategy."""

    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy) -> None:
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    pass