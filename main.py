"""
FastAPI app serving a scikit-learn model (Iris classifier) with logging.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
"""

import logging
import os
import sys
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging: print to stdout so Render's "Logs" tab shows everything
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ml-api")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Getting Started with ML in Production API",
    description="A minimal prediction API built for the workshop.",
    version="1.0.0",
)

# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.error("model.pkl not found at %s - /predict will return 503", MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width]"
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request: method, path, status code and how long it took."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.get("/")
def root():
    return {"message": "Workshop ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    """Basic health check endpoint - useful for deployment platforms & load balancers."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run train.py?")

    features = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(features)[0])
    class_name = IRIS_CLASSES[pred]

    logger.info("PREDICTION | features=%s -> %s (%d)", request.features, class_name, pred)
    return PredictionResponse(prediction=pred, class_name=class_name)
