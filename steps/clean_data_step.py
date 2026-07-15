from typing import List, Optional

import pandas as pd
from zenml import step

from src.data_cleaning import (
    DataCleaner,
    DropSpecifiedColumnsStrategy,
    MinValueRowFilterStrategy,
    PercentileOutlierRemovalStrategy,
)

DEFAULT_COLUMNS_TO_DROP = [
    "Address",
    "Latitude",
    "Longitude",
    "Rent_per_sqft",
    "Rent_category",
    "Frequency",
    "Purpose",
    "Posted_date",
    "Age_of_listing_in_days",
]

@step(enable_cache=False)
#@step
def clean_data_step(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    outlier_columns: Optional[List[str]] = None,
    outlier_percentile: float = 0.995,
) -> pd.DataFrame:
    columns_to_drop = columns_to_drop or DEFAULT_COLUMNS_TO_DROP
    outlier_columns = outlier_columns or ["Rent", "Area_in_sqft"]

    cleaner = (
        DataCleaner()
        .add_step(DropSpecifiedColumnsStrategy(columns_to_drop))
        .add_step(MinValueRowFilterStrategy(column="Rent", min_value=0, inclusive=True))
        .add_step(PercentileOutlierRemovalStrategy(upper_percentile=outlier_percentile))
    )

    return cleaner.run(df, outlier_columns=outlier_columns) 