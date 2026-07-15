from sklearn.base import RegressorMixin
from zenml import Model, step


@step
def model_loader(model_name: str) -> RegressorMixin:
    """
    Loads the current production model. Unlike an older version of this
    project, this loads a plain fitted regressor (e.g. RandomForestRegressor),
    NOT a sklearn Pipeline — preprocessing is handled separately by the
    persisted FeatureEngineer (backend/model/feature_engineer.pkl), not
    bundled into this artifact.
    """
    model = Model(name=model_name, version="production")
    trained_model: RegressorMixin = model.load_artifact("trained_model")
    return trained_model