---
title: UAE Rent Prediction API
emoji: 
sdk: docker
app_port: 7860
---

# UAE Rent Prediction API

FastAPI backend serving a trained Random Forest model that predicts annual property rent (AED) for UAE listings.

## Endpoints
- `GET /health` — health check, confirms model + feature engineer are loaded
- `POST /predict` — returns predicted annual rent given property features
- `GET /docs` 