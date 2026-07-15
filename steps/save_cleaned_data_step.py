import logging
from pathlib import Path

import pandas as pd
from zenml import step

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_OUTPUT_FILENAME = "cleaned_data.csv"


@step
def save_cleaned_data_step(
    cleaned_data: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = DEFAULT_OUTPUT_FILENAME,
) -> str:
    """
    Saves the cleaned dataset to a CSV file.

    Parameters
    ----------
    cleaned_data : pd.DataFrame
        The cleaned dataset to persist.
    output_dir : str
        Directory to save into (default: "data/processed").
    filename : str
        Output filename (default: "cleaned_data.csv").

    Returns
    -------
    str
        The full path the file was saved to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_path = output_path / filename
    cleaned_data.to_csv(full_path, index=False)

    logging.info(f"Cleaned dataset saved to {full_path} ({cleaned_data.shape[0]} rows, {cleaned_data.shape[1]} cols)")
    return str(full_path)