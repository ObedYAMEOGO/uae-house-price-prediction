from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from zenml import step

from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    OneHotEncoding,
    RareCategoryBucketingStrategy,
)

DEFAULT_PIPELINE_PATH = "backend/model/feature_engineer.pkl"


@step
def feature_engineering_step(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    location_min_count: int = 100,
    log_features: Optional[List[str]] = None,
    onehot_features: Optional[List[str]] = None,
    pipeline_save_path: str = DEFAULT_PIPELINE_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_features = log_features or ["Rent", "Area_in_sqft"]
    onehot_features = onehot_features or ["City", "Type", "Furnishing", "Location"]

    engineer = (
        FeatureEngineer()
        .add_step(RareCategoryBucketingStrategy(feature="Location", min_count=location_min_count))
        .add_step(LogTransformation(features=log_features))
        .add_step(OneHotEncoding(features=onehot_features))
    )

    train_transformed = engineer.fit_transform(train_df)
    test_transformed = engineer.transform(test_df)

    save_path = Path(pipeline_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    engineer.save(str(save_path))

    return train_transformed, test_transformed