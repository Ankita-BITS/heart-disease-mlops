import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "heart_disease_pipeline.joblib"

FEATURE_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


class HeartDiseaseInput(BaseModel):
    age: float = Field(..., example=63)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=1)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    heart_disease_probability: float
    confidence_probability: float


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


model = load_model()

app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI service for predicting heart disease risk using a trained ML pipeline.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        "request method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "predict_url": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartDiseaseInput):
    try:
        data = (
            input_data.model_dump()
            if hasattr(input_data, "model_dump")
            else input_data.dict()
        )

        input_df = pd.DataFrame([data])
        input_df = input_df[FEATURE_COLUMNS]

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        positive_class_index = classes.index(1)
        heart_disease_probability = float(probabilities[positive_class_index])

        if prediction == 1:
            confidence_probability = heart_disease_probability
            prediction_label = "heart_disease_present"
        else:
            confidence_probability = 1 - heart_disease_probability
            prediction_label = "heart_disease_absent"

        logger.info(
            "prediction_completed prediction=%s heart_disease_probability=%.4f",
            prediction,
            heart_disease_probability,
        )

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "heart_disease_probability": round(heart_disease_probability, 4),
            "confidence_probability": round(confidence_probability, 4),
        }

    except Exception as error:
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail=str(error)) from error