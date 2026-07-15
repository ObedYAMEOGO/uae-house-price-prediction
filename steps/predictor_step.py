# steps/predictor_step.py
import json
import logging
import numpy as np
import pandas as pd
import requests
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from src.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_FEATURE_ENGINEER_PATH = "backend/model/feature_engineer.pkl"
EXPECTED_RAW_COLUMNS = ["Beds", "Baths", "Type", "Area_in_sqft", "City", "Furnishing", "Location"]

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
    feature_engineer_path: str = DEFAULT_FEATURE_ENGINEER_PATH,
) -> np.ndarray:
    """
    Runs inference against the deployed MLflow model.
    """
    service.start(timeout=180)

    # Parse input data
    data = json.loads(input_data)
    data.pop("index", None)
    df = pd.DataFrame(data["data"], columns=data.get("columns", EXPECTED_RAW_COLUMNS))

    logging.info(f"Loading feature engineering pipeline from {feature_engineer_path}")
    feature_engineer = FeatureEngineer.load(feature_engineer_path)
    df_processed = feature_engineer.transform(df)

    logging.info(f"Sending {df_processed.shape[0]} rows, {df_processed.shape[1]} features")

    # Try using the service first
    try:
        logging.info("Attempting prediction through service...")
        prediction_log_space = service.predict(df_processed)
        logging.info("Service prediction succeeded!")
    except Exception as e:
        logging.warning(f"Service prediction failed: {e}")
        
        # Fallback: Direct HTTP request
        logging.info("Using direct HTTP request to MLflow server...")
        
        # Get the endpoint from the service
        endpoint = service.prediction_url
        logging.info(f"Endpoint: {endpoint}")
        
        # Try different payload formats
        payload_formats = [
            {"inputs": df_processed.to_dict(orient="records")},
            {"instances": df_processed.to_dict(orient="records")},
            {"dataframe_records": df_processed.to_dict(orient="records")},
        ]
        
        for payload in payload_formats:
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction_log_space = np.array(result["predictions"])
                    logging.info(f"HTTP request succeeded with payload format: {list(payload.keys())[0]}")
                    break
                else:
                    logging.warning(f"HTTP request failed with status {response.status_code}")
                    logging.warning(f"Response: {response.text}")
            except Exception as e2:
                logging.warning(f"HTTP request failed: {e2}")
                continue
        else:
            raise RuntimeError("All prediction methods failed")

    # Convert from log space back to actual AED
    prediction_aed = np.expm1(prediction_log_space)
    logging.info(f"Predictions (AED): {prediction_aed}")
    
    return prediction_aed