# steps/dynamic_data_importer_step.py
import json
import logging
import pandas as pd
from zenml import step

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@step
def dynamic_importer() -> str:
    """
    Simulates incoming raw property listings for batch inference.
    """
    data = {
        "Beds": [1, 2, 3, 4, 5],
        "Baths": [1, 2, 3, 4, 5],
        "Type": ["Apartment", "Apartment", "Villa", "Villa", "Penthouse"],
        "Area_in_sqft": [800, 1200, 2500, 3500, 4500],
        "City": ["Dubai", "Dubai", "Abu Dhabi", "Abu Dhabi", "Dubai"],
        "Furnishing": ["Unfurnished", "Furnished", "Unfurnished", "Furnished", "Furnished"],
        "Location": ["Dubai Marina", "JLT", "Al Raha Beach", "Yas Island", "Downtown Dubai"],
    }
    df = pd.DataFrame(data)
    
    logging.info(f"Sample data for inference: {df.to_dict()}")
    return df.to_json(orient="split")