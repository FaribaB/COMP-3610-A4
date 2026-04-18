import importlib
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


NUMERIC_FEATURES = [
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
    "trip_speed_mph",
    "log_trip_distance",
    "fare_per_mile",
    "fare_per_minute",
    "passenger_count",
    "fare_amount",
]

CATEGORICAL_FEATURES = ["PU_Borough", "DO_Borough"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_test_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = DummyRegressor(strategy="constant", constant=4.25)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    X = pd.DataFrame([
        {
            "trip_distance": 1.0,
            "pickup_hour": 10,
            "pickup_day_of_week": 2,
            "is_weekend": False,
            "trip_duration_minutes": 12.0,
            "trip_speed_mph": 5.0,
            "log_trip_distance": 0.69,
            "fare_per_mile": 6.0,
            "fare_per_minute": 0.5,
            "passenger_count": 1,
            "fare_amount": 12.0,
            "PU_Borough": "Manhattan",
            "DO_Borough": "Queens",
        },
        {
            "trip_distance": 2.0,
            "pickup_hour": 18,
            "pickup_day_of_week": 4,
            "is_weekend": False,
            "trip_duration_minutes": 20.0,
            "trip_speed_mph": 6.0,
            "log_trip_distance": 1.10,
            "fare_per_mile": 7.0,
            "fare_per_minute": 0.7,
            "passenger_count": 2,
            "fare_amount": 18.0,
            "PU_Borough": "Brooklyn",
            "DO_Borough": "Manhattan",
        },
    ])
    y = [4.25, 4.25]

    pipe.fit(X, y)
    return pipe


def configure_app(tmp_path):
    model = build_test_pipeline()

    model_path = tmp_path / "rfreg_pipe.pkl"
    metadata_path = tmp_path / "model_metadata.json"

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps({
        "model_name": "rfreg-pipeline-test-model",
        "version": "1.0.0",
        "features": FEATURES,
        "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.5},
        "trained_date": "2026-04-16"
    }))

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["METADATA_PATH"] = str(metadata_path)

    # Clear legacy scaler env var if app.py still checks for it
    os.environ.pop("SCALER_PATH", None)

    if "app" in sys.modules:
        del sys.modules["app"]

    import app
    importlib.reload(app)
    return app


def valid_payload():
    return {
        "trip_distance": 3.5,
        "pickup_hour": 18,
        "pickup_day_of_week": 4,
        "trip_duration_minutes": 15.0,
        "fare_amount": 18.5,
        "pu_borough": "Manhattan",
        "do_borough": "Queens",
    }


def test_root(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_health(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_predict_valid(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=valid_payload())
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 4.25
        assert "prediction_id" in data


def test_batch_prediction(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        payload = {"records": [valid_payload(), valid_payload(), valid_payload()]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["count"] == 3


def test_predict_missing_field(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        payload = valid_payload()
        del payload["fare_amount"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_predict_invalid_type(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        payload = valid_payload()
        payload["trip_distance"] = "bad"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_predict_out_of_range(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        payload = valid_payload()
        payload["pickup_hour"] = 30
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_zero_distance_rejected(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        payload = valid_payload()
        payload["trip_distance"] = 0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_model_info(tmp_path):
    from fastapi.testclient import TestClient
    app_module = configure_app(tmp_path)
    with TestClient(app_module.app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "metrics" in data