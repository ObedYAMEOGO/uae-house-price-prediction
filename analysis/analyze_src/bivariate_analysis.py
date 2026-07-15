from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BivariateAnalysisStrategy(ABC):
    """Strategy interface for analyzing the relationship between two features."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Analyze and visualize the relationship between two features."""
        pass


class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """Scatter plot for two numerical features."""

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """Box plot for a categorical feature against a numerical target."""

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalyzer:
    """Context class: delegates two-feature analysis to a chosen strategy."""

    def __init__(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        self._strategy.analyze(df, feature1, feature2)


if __name__ == "__main__":
    pass