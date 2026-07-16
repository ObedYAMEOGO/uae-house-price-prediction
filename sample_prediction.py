# sample_prediction.py
import json
import numpy as np
import pandas as pd
import requests
from src.feature_engineering import FeatureEngineer

MLFLOW_ENDPOINT_URL = "http://127.0.0.1:8000/invocations"
FEATURE_ENGINEER_PATH = "backend/model/feature_engineer.pkl"

# Raw property data, in the same schema the model saw before feature engineering
raw_input = pd.DataFrame([
    {
        "Beds": 2,
        "Baths": 2,
        "Type": "Apartment",
        "Area_in_sqft": 1200,
        "City": "Dubai",
        "Furnishing": "Unfurnished",
        "Location": "Dubai Marina",
    },
    {
        "Beds": 4,
        "Baths": 3,
        "Type": "Villa",
        "Area_in_sqft": 3500,
        "City": "Abu Dhabi",
        "Furnishing": "Furnished",
        "Location": "Al Raha Beach",
    },
])

print("Loading feature engineering pipeline...")
feature_engineer = FeatureEngineer.load(FEATURE_ENGINEER_PATH)

print("Applying feature engineering (bucketing, one-hot encoding)...")
processed_input = feature_engineer.transform(raw_input)

print(f"Processed data shape: {processed_input.shape}")
print(f"Processed columns: {processed_input.columns.tolist()[:5]}... (showing first 5)")

payload_formats = [
    {
        "dataframe_split": {
            "columns": processed_input.columns.tolist(),
            "data": processed_input.values.tolist(),
        }
    },
    {
        "inputs": processed_input.to_dict(orient="records")
    },
    {
        "instances": processed_input.to_dict(orient="records")
    },
]

headers = {"Content-Type": "application/json"}

# Try each format until one works
for idx, payload in enumerate(payload_formats, 1):
    format_name = list(payload.keys())[0]
    print(f"\nTrying format {idx}: {format_name}")
    
    try:
        response = requests.post(
            MLFLOW_ENDPOINT_URL, 
            headers=headers, 
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions_log_space = result.get("predictions", result)
            
            print(f"\n SUCCESS with format: {format_name}")
            print("\nRaw predictions (log-space):", predictions_log_space)
            
            # Convert from log space back to actual AED
            predictions_aed = np.expm1(predictions_log_space)
            print("Predicted Rent (AED):", predictions_aed)
            
            # Print individual predictions
            for i, pred in enumerate(predictions_aed):
                print(f"  Property {i+1}: AED {pred:,.2f}")
            
            break
        else:
            print(f" Failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f" Error with format {format_name}: {e}")
else:
    print("\n All payload formats failed. Check if the MLflow service is running.")
    print(f"Try: curl -X GET {MLFLOW_ENDPOINT_URL.replace('/invocations', '/ping')}")