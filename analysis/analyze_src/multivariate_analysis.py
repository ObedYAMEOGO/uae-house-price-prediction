from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    """
    Template Method pattern: defines the fixed skeleton for multivariate
    analysis (correlation heatmap -> pairplot).
    """

    def analyze(self, df: pd.DataFrame) -> None:
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame) -> None:
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    """Concrete implementation for numerical-feature correlation + pairplot."""

    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=["number"])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        sns.pairplot(df.select_dtypes(include=["number"]))
        plt.suptitle("Pair Plot of Numerical Features", y=1.02)
        plt.show()


if __name__ == "__main__":
    pass