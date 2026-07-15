from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class OutlierDetectionStrategy(ABC):
    """Strategy interface for detecting outliers in a numerical column."""

    @abstractmethod
    def detect(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Returns a boolean Series flagging outlier rows for the given column."""
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """Flags values outside [Q1 - k*IQR, Q3 + k*IQR]."""

    def __init__(self, k: float = 1.5) -> None:
        self.k = k

    def detect(self, df: pd.DataFrame, column: str) -> pd.Series:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        return (df[column] < lower) | (df[column] > upper)


class PercentileOutlierDetection(OutlierDetectionStrategy):
    """Flags values above a given upper percentile threshold (good for heavy right-skew)."""

    def __init__(self, upper_percentile: float = 0.995) -> None:
        self.upper_percentile = upper_percentile

    def detect(self, df: pd.DataFrame, column: str) -> pd.Series:
        threshold = df[column].quantile(self.upper_percentile)
        return df[column] > threshold


class OutlierAnalyzer:
    """Context class: delegates outlier detection to a chosen strategy and reports/plots results."""

    def __init__(self, strategy: OutlierDetectionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy) -> None:
        self._strategy = strategy

    def analyze(self, df: pd.DataFrame, column: str) -> pd.Series:
        outlier_mask = self._strategy.detect(df, column)
        print(f"\n{column}: {outlier_mask.sum()} outliers detected ({outlier_mask.mean()*100:.2f}% of rows)")

        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()

        return outlier_mask


if __name__ == "__main__":
    pass