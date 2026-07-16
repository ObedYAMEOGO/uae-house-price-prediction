import logging
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.base import RegressorMixin

from feature_engineering import FeatureEngineer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_ENGINEER_PATH = MODEL_DIR / "feature_engineer.pkl"


@lru_cache(maxsize=1)
def load_model() -> RegressorMixin:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logging.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_engineer() -> FeatureEngineer:
    if not FEATURE_ENGINEER_PATH.exists():
        raise FileNotFoundError(f"Feature engineer not found at {FEATURE_ENGINEER_PATH}")
    logging.info(f"Loading feature engineer from {FEATURE_ENGINEER_PATH}")
    return FeatureEngineer.load(str(FEATURE_ENGINEER_PATH))