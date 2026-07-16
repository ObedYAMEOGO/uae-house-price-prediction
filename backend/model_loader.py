import logging
import urllib.request
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.base import RegressorMixin

from feature_engineering import FeatureEngineer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_ENGINEER_PATH = MODEL_DIR / "feature_engineer.pkl"

MODEL_URL = "https://huggingface.co/spaces/thehatbuddy/uae-rent-prediction-api/resolve/main/backend/model/model.pkl"
FEATURE_ENGINEER_URL = "https://huggingface.co/spaces/thehatbuddy/uae-rent-prediction-api/resolve/main/backend/model/feature_engineer.pkl"


def _download_if_needed(path: Path, url: str, min_size_bytes: int = 1000) -> None:
    """
    Downloads the artifact if it's missing OR if what's on disk is
    suspiciously small (e.g. a Git LFS pointer file left over from a
    checkout that didn't resolve LFS content, rather than the real binary).
    """
    if path.exists() and path.stat().st_size > min_size_bytes:
        return

    logging.info(f"Downloading {path.name} from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    logging.info(f"Downloaded {path.name} ({path.stat().st_size} bytes)")


@lru_cache(maxsize=1)
def load_model() -> RegressorMixin:
    _download_if_needed(MODEL_PATH, MODEL_URL)
    logging.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_engineer() -> FeatureEngineer:
    _download_if_needed(FEATURE_ENGINEER_PATH, FEATURE_ENGINEER_URL)
    logging.info(f"Loading feature engineer from {FEATURE_ENGINEER_PATH}")
    return FeatureEngineer.load(str(FEATURE_ENGINEER_PATH))