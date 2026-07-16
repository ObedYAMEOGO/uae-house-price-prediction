import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#laod model
from model_loader import load_model, load_feature_engineer
from routes.predict import router as predict_router
from schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="UAE Rent Prediction API",
    description="Predicts annual property rent (AED) for UAE listings based on property features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    model_ok = True
    fe_ok = True
    try:
        load_model()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        model_ok = False
    
    try:
        load_feature_engineer()
    except Exception as e:
        logging.error(f"Failed to load feature engineer: {e}")
        fe_ok = False

    return HealthResponse(
        status="ok",
        model_loaded=model_ok,
        feature_engineer_loaded=fe_ok,
    )