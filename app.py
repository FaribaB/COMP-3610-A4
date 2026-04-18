import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/rfreg_pipe.pkl"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "models/model_metadata.json"))

ALL_BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown"]

DEFAULT_MODEL_FEATURES = [
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
    "trip_speed_mph",
    "log_trip_distance",
    "fare_per_mile",
    "fare_per_minute",
    "PU_Borough",
    "DO_Borough",
    "passenger_count",
    "trip_distance",
    "fare_amount",
]

ml_model: Any = None
start_time: Optional[float] = None
model_metadata: Dict[str, Any] = {}
feature_columns: List[str] = DEFAULT_MODEL_FEATURES.copy()


class TaxiTripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, le=200, description="Trip distance in miles")
    pickup_hour: int = Field(..., ge=0, le=23, description="Pickup hour of day")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    trip_duration_minutes: float = Field(..., gt=0, le=600, description="Trip duration in minutes")
    fare_amount: float = Field(..., gt=0, le=500, description="Metered fare amount in USD")
    pu_borough: str = Field(..., description="Pickup borough")
    do_borough: str = Field(..., description="Dropoff borough")
    passenger_count: int = Field(1, ge=1, le=8, description="Passenger count")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trip_distance": 3.5,
                    "pickup_hour": 18,
                    "pickup_day_of_week": 4,
                    "trip_duration_minutes": 15.0,
                    "fare_amount": 18.5,
                    "pu_borough": "Manhattan",
                    "do_borough": "Queens",
                    "passenger_count": 1,
                }
            ]
        }
    }

    @field_validator("pu_borough", "do_borough")
    @classmethod
    def validate_borough(cls, value: str) -> str:
        if value not in ALL_BOROUGHS:
            raise ValueError(f"Borough must be one of: {', '.join(ALL_BOROUGHS)}")
        return value


class PredictionResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str


class BatchInput(BaseModel):
    records: List[TaxiTripFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


def load_metadata() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())

    return {
        "model_name": MODEL_PATH.stem,
        "version": "1.0.0",
        "features": feature_columns,
        "metrics": {"mae": None, "rmse": None, "r2": None},
        "trained_date": None,
        "notes": "Metadata file not found. Add model_metadata.json if required.",
    }


def build_feature_frame(record: TaxiTripFeatures) -> pd.DataFrame:
    is_weekend = bool(record.pickup_day_of_week in [5, 6])
    trip_speed_mph = 60.0 * record.trip_distance / record.trip_duration_minutes
    log_trip_distance = float(np.log1p(record.trip_distance))
    fare_per_mile = record.fare_amount / record.trip_distance
    fare_per_minute = record.fare_amount / record.trip_duration_minutes

    row = {
        "pickup_hour": record.pickup_hour,
        "pickup_day_of_week": record.pickup_day_of_week,
        "is_weekend": is_weekend,
        "trip_duration_minutes": record.trip_duration_minutes,
        "trip_speed_mph": trip_speed_mph,
        "log_trip_distance": log_trip_distance,
        "fare_per_mile": fare_per_mile,
        "fare_per_minute": fare_per_minute,
        "PU_Borough": record.pu_borough,
        "DO_Borough": record.do_borough,
        "passenger_count": record.passenger_count,
        "trip_distance": record.trip_distance,
        "fare_amount": record.fare_amount,
    }

    df = pd.DataFrame([row])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    return df[feature_columns]


def model_predict(record: TaxiTripFeatures) -> float:
    features_df = build_feature_frame(record)
    prediction = ml_model.predict(features_df)[0]
    return round(float(prediction), 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, start_time, model_metadata, feature_columns

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Expected pipeline model: rfreg_pipe.pkl"
        )

    ml_model = joblib.load(MODEL_PATH)

    model_metadata = load_metadata()
    if model_metadata.get("features"):
        feature_columns = model_metadata["features"]
    else:
        feature_columns = DEFAULT_MODEL_FEATURES.copy()
        model_metadata["features"] = feature_columns

    start_time = time.time()
    yield


app = FastAPI(
    title="Taxi Tip Prediction API",
    version="1.0.0",
    description="FastAPI service for predicting NYC taxi tip amounts using rfreg_pipe.pkl.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Taxi Tip Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TaxiTripFeatures) -> PredictionResponse:
    prediction = model_predict(input_data)
    version = str(model_metadata.get("version", "1.0.0"))
    return PredictionResponse(
        prediction=prediction,
        prediction_id=str(uuid.uuid4()),
        model_version=version,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput) -> BatchResponse:
    started = time.time()
    preds = [
        PredictionResponse(
            prediction=model_predict(record),
            prediction_id=str(uuid.uuid4()),
            model_version=str(model_metadata.get("version", "1.0.0")),
        )
        for record in batch.records
    ]
    return BatchResponse(
        predictions=preds,
        count=len(preds),
        processing_time_ms=round((time.time() - started) * 1000, 2),
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": str(model_metadata.get("version", "1.0.0")),
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0.0,
    }


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    return {
        "model_name": model_metadata.get("model_name", MODEL_PATH.stem),
        "version": str(model_metadata.get("version", "1.0.0")),
        "features": model_metadata.get("features", feature_columns),
        "metrics": model_metadata.get("metrics", {"mae": None, "rmse": None, "r2": None}),
        "trained_date": model_metadata.get("trained_date"),
        "source_model_path": str(MODEL_PATH),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )