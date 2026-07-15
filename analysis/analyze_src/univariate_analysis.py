from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class UnivariateAnalysisStrategy(ABC):
    """Strategy interface for analyzing a single feature."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Analyze and visualize a single feature's distribution."""
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """Histogram + KDE for a numerical feature."""

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """Count plot for a categorical feature."""

    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x=feature, data=df, palette="muted",
            order=df[feature].value_counts().index,
        )
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    """Context class: delegates single-feature analysis to a chosen strategy."""

    def __init__(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str) -> None:
        self._strategy.analyze(df, feature)


if __name__ == "__main__":
    pass