from abc import ABC, abstractmethod

import pandas as pd


class DataInspectionStrategy(ABC):
    """Strategy interface for different kinds of basic data inspection."""

    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Perform a specific type of inspection on the dataframe."""
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    """Inspects column dtypes and non-null counts."""

    def inspect(self, df: pd.DataFrame) -> None:
        print("\nData Types and Non-Null Counts:")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    """Inspects summary statistics for numerical and categorical columns."""

    def inspect(self, df: pd.DataFrame) -> None:
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())

        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["object", "category"]))
        
        
# I added this strategy after inspecting the missing values to check properly if we could perform the localized neighborhood imputation
class MissingCoordinatesNeighborStrategy(DataInspectionStrategy):
    """
    Inspects locations with missing geographical coordinates to 
    determine if valid neighbor data exists for imputation.
    """

    def inspect(self, df: pd.DataFrame) -> None:
        print("\nChecking for Imputation Viability (Missing Coordinates):")
        
        # 1. Identify locations missing coordinates
        missing_coords_locations = df[df['Latitude'].isnull()]['Location'].unique()
        
        if len(missing_coords_locations) == 0:
            print("No missing coordinates found. Dataset is clean.")
            return

        # 2. Filter dataset for these locations
        affected_df = df[df['Location'].isin(missing_coords_locations)]
        
        # 3. Count valid coordinates per location
        neighbor_counts = affected_df.groupby('Location')['Latitude'].count().reset_index()
        neighbor_counts.columns = ['Location', 'Valid_Neighbors_Available']
        
        # 4. Output the results
        print(neighbor_counts.to_string(index=False))

class DataInspector:
    """
    Context class: holds a reference to a DataInspectionStrategy and
    delegates the inspection to it. Lets you swap strategies at runtime.
    """

    def __init__(self, strategy: DataInspectionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        """Swap the inspection strategy at runtime."""
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> None:
        self._strategy.inspect(df)


if __name__ == "__main__":
    pass