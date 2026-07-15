# src/data_ingestion.py

import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataIngestor(ABC):
    """Abstract base class defining the blueprint for data ingestion strategies."""

    @abstractmethod
    def ingest(self, file_path: Path, extract_to: Path) -> pd.DataFrame:
        """
        Extracts and loads data from a file into a Pandas DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the source data file (e.g. a .zip archive).
        extract_to : Path
            Directory where extracted contents should be written.

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError("Subclasses must implement the ingest method.")


class ZipFileDataIngestor(DataIngestor):
    """
    Ingestor for extracting and reading a CSV file from a ZIP archive.

    Use when the raw dataset is distributed as a .zip (e.g. a Kaggle download)
    containing at least one CSV file.
    """

    def ingest(self, file_path: Path, extract_to: Path) -> pd.DataFrame:
        if file_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip file, got: {file_path.name}")

        if not file_path.exists():
            raise FileNotFoundError(f"Archive not found at: {file_path}")

        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            csv_names = [name for name in zip_ref.namelist() if name.lower().endswith(".csv")]

            if not csv_names:
                raise FileNotFoundError("No CSV file found inside the ZIP archive.")
            if len(csv_names) > 1:
                print(
                    f"Warning: {len(csv_names)} CSV files found in archive "
                    f"({csv_names}). Using the first one: '{csv_names[0]}'."
                )

            zip_ref.extractall(extract_to)
            csv_path = extract_to / csv_names[0]

        df = pd.read_csv(csv_path)
        print(f"Successfully loaded '{csv_names[0]}' ({df.shape[0]} rows, {df.shape[1]} cols) into a DataFrame.")
        return df


class DataIngestorFactory:
    """Factory that selects the appropriate DataIngestor based on file extension."""

    @staticmethod
    def get_data_ingestor(file_path: Path) -> DataIngestor:
        """
        Returns the correct DataIngestor implementation for the given file type.

        Extend this method (add an elif branch + new DataIngestor subclass)
        to support additional formats later, e.g. .tar.gz or .json.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".zip":
            return ZipFileDataIngestor()

        raise ValueError(f"Unsupported file type: '{suffix}'. Only .zip is currently supported.")


if __name__ == "__main__":
    pass