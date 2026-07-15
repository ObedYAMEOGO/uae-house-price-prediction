from typing import Tuple

import pandas as pd
from zenml import step

from src.data_splitting import DataSplitter, StratifiedSplitStrategy # type: ignore



@step
def data_splitter_step(
    df: pd.DataFrame,
    stratify_column: str = "City",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the cleaned dataset into training and testing sets, stratified by
    City so each city keeps its proportional representation across both
    splits (Dubai has ~34,250 rows vs. Umm Al Quwain's ~65 — a plain random
    split risks under- or over-representing small cities in the test set).

    Returns train_df and test_df with the target column (Rent) still
    included in both — feature engineering (which needs to log-transform
    Rent) runs next, and X/y separation happens only after that.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset (post data_cleaning step).
    stratify_column : str
        Column to stratify the split on (default: "City").
    test_size : float
        Proportion of data held out for testing (default: 0.2).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    splitter = DataSplitter(
        strategy=StratifiedSplitStrategy(stratify_column=stratify_column, test_size=test_size)
    )
    train_df, test_df = splitter.split(df)
    return train_df, test_df