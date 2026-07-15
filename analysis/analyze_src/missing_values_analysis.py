from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    """
    Template Method pattern: defines the fixed skeleton for missing-values
    analysis (report -> visualize), while subclasses customize each step.
    """

    def analyze(self, df: pd.DataFrame) -> None:
        """
        Runs the full missing-values analysis: report, then plot.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to analyze.
        """
        self.report_missing_values(df)
        self.plot_missing_values(df)

    @abstractmethod
    def report_missing_values(self, df: pd.DataFrame) -> None:
        """Print/log a summary of missing values. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def plot_missing_values(self, df: pd.DataFrame) -> None:
        """Visualize missing value patterns. Must be implemented by subclasses."""
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    """Concrete implementation: console summary + heatmap."""

    def report_missing_values(self, df: pd.DataFrame) -> None:
        print("\nMissing Values Summary:")
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            print("No missing values detected.")
        else:
            print(missing_counts)

    def plot_missing_values(self, df: pd.DataFrame) -> None:
        if df.isnull().sum().sum() == 0:
            print("\nNo missing values detected. Skipping heatmap generation.")
            return

        print("\nGenerating Missing Values Heatmap...")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="plasma")
        plt.title("Missing Values Heatmap")
        plt.show()


if __name__ == "__main__":
    pass