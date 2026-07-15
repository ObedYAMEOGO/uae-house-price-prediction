import logging
from pathlib import Path

import joblib
from sklearn.base import RegressorMixin
from zenml import step

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_MODEL_SAVE_PATH = "backend/model/model.pkl"


@step
def register_model_step(
    trained_model: RegressorMixin,
    should_register: bool = True,
    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
) -> str:
    """
    Exports the trained model to disk as a standalone artifact, so the
    FastAPI backend can load it directly via joblib — completely independent
    of ZenML or MLflow's serving infrastructure.

    This is the model-side counterpart to feature_engineering_step's
    persisted FeatureEngineer (backend/model/feature_engineer.pkl). At
    inference time, the backend loads both and applies them in sequence:
        1. feature_engineer.transform(raw_input)
        2. model.predict(processed_input)

    Parameters
    ----------
    trained_model : RegressorMixin
        The trained model (output of model_building_step).
    should_register : bool
        Whether to actually save the model — lets this be wired to the same
        kind of quality-gate decision as deployment_trigger_step, so a
        poorly performing model doesn't silently overwrite a good one.
    model_save_path : str
        Where to save the model artifact.

    Returns
    -------
    str
        The path the model was saved to (or a message indicating it was
        skipped).
    """
    if not should_register:
        logging.info("should_register=False — skipping model export.")
        return "Model registration skipped."

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model, save_path)
    logging.info(f"Model exported to {save_path}")

    return str(save_path)