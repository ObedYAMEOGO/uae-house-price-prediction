import logging
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.base import RegressorMixin

from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_ENGINEER_PATH = MODEL_DIR / "feature_engineer.pkl"


@lru_cache(maxsize=1)
def load_model() -> RegressorMixin:
    """Loads the trained model once and caches it for the process lifetime."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model artifact found at {MODEL_PATH}. Run the training pipeline "
            f"(python run_pipeline.py) to generate it first."
        )
    logging.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_engineer() -> FeatureEngineer:
    """Loads the fitted feature engineering pipeline once and caches it."""
    if not FEATURE_ENGINEER_PATH.exists():
        raise FileNotFoundError(
            f"No feature engineer artifact found at {FEATURE_ENGINEER_PATH}. "
            f"Run the training pipeline first."
        )
    logging.info(f"Loading feature engineer from {FEATURE_ENGINEER_PATH}")
    return FeatureEngineer.load(str(FEATURE_ENGINEER_PATH))