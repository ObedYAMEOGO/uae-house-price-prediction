import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# 1. Column Dropping — remove leakage, constant, and irrelevant columns
# ============================================================
class ColumnDropStrategy(ABC):
    """Abstract base class for strategies that remove entire columns."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame with the relevant columns removed."""
        raise NotImplementedError("Subclasses must implement the apply method.")


class DropSpecifiedColumnsStrategy(ColumnDropStrategy):
    """
    Drops an explicit list of columns.

    Use for columns confirmed via EDA to be leakage (derived from the target),
    constant (zero variance), or too high-cardinality/irrelevant to use directly.
    """

    def __init__(self, columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns_to_drop if c in df.columns]
        missing = [c for c in self.columns_to_drop if c not in df.columns]

        if missing:
            logging.warning(f"Columns not found in DataFrame, skipping: {missing}")

        logging.info(f"Dropping columns: {existing}")
        return df.drop(columns=existing)


# ============================================================
# 2. Invalid Row Filtering — remove rows that are structurally invalid
# ============================================================
class RowFilterStrategy(ABC):
    """Abstract base class for strategies that remove invalid rows."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the apply method.")


class MinValueRowFilterStrategy(RowFilterStrategy):
    """
    Drops rows where a given column's value is less than or equal to a minimum threshold.

    Use for: e.g. Rent <= 0, which represents invalid/placeholder listings rather
    than genuine zero-rent properties.
    """

    def __init__(self, column: str, min_value: float = 0, inclusive: bool = True):
        self.column = column
        self.min_value = min_value
        self.inclusive = inclusive

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if self.inclusive:
            df_filtered = df[df[self.column] > self.min_value]
        else:
            df_filtered = df[df[self.column] >= self.min_value]

        removed = before - len(df_filtered)
        logging.info(
            f"Removed {removed} rows where {self.column} <= {self.min_value} "
            f"({removed / before * 100:.2f}% of data)"
        )
        return df_filtered


# ============================================================
# 3. Outlier Handling — cap or remove extreme values
# ============================================================
class OutlierHandlingStrategy(ABC):
    """Abstract base class for strategies that handle outliers in a numerical column."""

    @abstractmethod
    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the apply method.")


class PercentileOutlierRemovalStrategy(OutlierHandlingStrategy):
    """
    Removes rows above a given upper percentile threshold.

    Preferred over IQR for heavily right-skewed columns (e.g. Rent, Area_in_sqft
    in this dataset both have skewness > 18), since IQR would flag far too many
    legitimately high-but-valid observations as outliers.
    """

    def __init__(self, upper_percentile: float = 0.995):
        self.upper_percentile = upper_percentile

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        threshold = df[column].quantile(self.upper_percentile)
        before = len(df)
        df_filtered = df[df[column] <= threshold]
        removed = before - len(df_filtered)

        logging.info(
            f"Removed {removed} outlier rows from '{column}' "
            f"(above {self.upper_percentile*100:.1f}th percentile = {threshold:.2f})"
        )
        return df_filtered


# ============================================================
# 4. Missing Value Handling — general-purpose, reusable for future datasets
# ============================================================
class NaNValuesHandlingStrategy(ABC):
    """Abstract base class for handling missing values in a DataFrame."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the handle method.")


class DropNaNValuesStrategy(NaNValuesHandlingStrategy):
    """Drops rows or columns containing NaN values."""

    def __init__(self, axis: int = 0, thresh: Optional[int] = None):
        """
        :param axis: 0 to drop rows with NaNs, 1 to drop columns with NaNs.
        :param thresh: Minimum number of non-NaN values required to keep the row/column.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis}, thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info(f"Rows before: {len(df)}, after: {len(df_cleaned)}")
        return df_cleaned


class FillNaNValuesStrategy(NaNValuesHandlingStrategy):
    """Fills NaN values using mean, median, mode, or a constant."""

    def __init__(self, method: str = "mean", fill_value=None):
        self.method = method.lower()
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")
        df_cleaned = df.copy()

        if self.method in ("mean", "median"):
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            stat = df_cleaned[numeric_columns].mean() if self.method == "mean" else df_cleaned[numeric_columns].median()
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(stat)

        elif self.method == "mode":
            for column in df_cleaned.columns:
                mode_series = df_cleaned[column].mode()
                if not mode_series.empty:
                    # Reassignment, not inplace-on-a-slice, to avoid SettingWithCopyWarning
                    df_cleaned[column] = df_cleaned[column].fillna(mode_series.iloc[0])

        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)

        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        return df_cleaned


class GroupedMedianFillStrategy(NaNValuesHandlingStrategy):
    """
    Fills NaN values in a target column using the median of that column,
    computed separately within each group of a categorical column.

    Example use case (not needed for this project since Latitude/Longitude
    are being dropped, but kept as reusable infrastructure): filling missing
    coordinates using the median coordinate of properties in the same City.
    """

    def __init__(self, target_column: str, group_column: str):
        self.target_column = target_column
        self.group_column = group_column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        group_medians = df_cleaned.groupby(self.group_column)[self.target_column].transform("median")
        df_cleaned[self.target_column] = df_cleaned[self.target_column].fillna(group_medians)
        logging.info(f"Filled '{self.target_column}' NaNs using median grouped by '{self.group_column}'.")
        return df_cleaned


# ============================================================
# Context classes — allow swapping strategies at runtime
# ============================================================
class DataCleaner:
    """
    Generic context class that runs a sequence of cleaning strategies
    (column drops, row filters, outlier handling) over a DataFrame.
    """

    def __init__(self):
        self._steps: List = []

    def add_step(self, strategy) -> "DataCleaner":
        """Adds a strategy to the pipeline. Chainable."""
        self._steps.append(strategy)
        return self

    def run(self, df: pd.DataFrame, outlier_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executes all added steps in order. Strategies with an `apply(df, column)`
        signature (like outlier handling) are applied once per column in
        `outlier_columns`; others use `apply(df)`.
        """
        for step in self._steps:
            if isinstance(step, OutlierHandlingStrategy):
                for column in (outlier_columns or []):
                    df = step.apply(df, column)
            else:
                df = step.apply(df)
        return df


class NaNValueHandler:
    """Context class for missing-value handling strategies (kept separate,
    since it operates via `.handle()` rather than `.apply()`)."""

    def __init__(self, strategy: NaNValuesHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: NaNValuesHandlingStrategy) -> None:
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)

if __name__ == "__main__":
    pass