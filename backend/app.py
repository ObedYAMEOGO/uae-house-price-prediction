from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.model_loader import load_model, load_feature_engineer
from backend.routes.predict import router as predict_router
from backend.schemas import HealthResponse

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
    except FileNotFoundError:
        model_ok = False
    try:
        load_feature_engineer()
    except FileNotFoundError:
        fe_ok = False

    return HealthResponse(
        status="ok" if (model_ok and fe_ok) else "degraded",
        model_loaded=model_ok,
        feature_engineer_loaded=fe_ok,
    )