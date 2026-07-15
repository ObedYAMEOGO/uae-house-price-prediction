from typing import Tuple

import pandas as pd
from zenml import step

from src.data_splitting import split_features_and_target


@step
def feature_target_split_step(
    train_transformed: pd.DataFrame,
    test_transformed: pd.DataFrame,
    target_column: str = "Rent",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates features (X) from the target (y), now that feature engineering
    (including log-transforming Rent) has already been applied.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_train, y_train = split_features_and_target(train_transformed, target_column)
    X_test, y_test = split_features_and_target(test_transformed, target_column)
    return X_train, X_test, y_train, y_test