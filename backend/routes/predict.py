import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from schemas import PropertyFeatures, PredictionResponse
from model_loader import load_model, load_feature_engineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

router = APIRouter()

# Maps the API's snake_case field names to the raw column names the
# FeatureEngineer/model were actually trained on.
FIELD_TO_COLUMN = {
    "beds": "Beds",
    "baths": "Baths",
    "type": "Type",
    "area_in_sqft": "Area_in_sqft",
    "city": "City",
    "furnishing": "Furnishing",
    "location": "Location",
}


@router.post("/predict", response_model=PredictionResponse)
def predict(features: PropertyFeatures):
    try:
        model = load_model()
        feature_engineer = load_feature_engineer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw_dict = {FIELD_TO_COLUMN[k]: [v] for k, v in features.model_dump().items()}
    raw_df = pd.DataFrame(raw_dict)

    try:
        processed_df = feature_engineer.transform(raw_df)
        prediction_log_space = model.predict(processed_df)
        prediction_aed = float(np.expm1(prediction_log_space[0]))
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(predicted_rent_aed=round(prediction_aed, 2))