from pathlib import Path

import pandas as pd
from zenml import step

from src.data_ingestion import DataIngestorFactory


@step
def data_ingestion_step(file_path: str, extract_to: str = "data/raw") -> pd.DataFrame:
    """
    ZenML step that extracts and loads the raw dataset from a ZIP archive.

    Parameters
    ----------
    file_path : str
        Path to the source .zip archive (e.g. "data/raw/archive.zip").
    extract_to : str
        Directory to extract the CSV into (default: "data/raw").

    Returns
    -------
    pd.DataFrame
    """
    archive_path = Path(file_path)
    extract_dir = Path(extract_to)

    data_ingestor = DataIngestorFactory.get_data_ingestor(archive_path)
    df = data_ingestor.ingest(archive_path, extract_dir)

    return df