# COMP 3610 Assignment 4 
## Overview

This project deploys a machine learning model for **NYC taxi tip prediction** as a production-style REST API using **FastAPI**. The workflow for the assignment includes:

- experiment tracking with **MLflow**
- model serving with **FastAPI**
- automated API testing with **pytest**
- containerization with **Docker** and **Docker Compose**

The deployed model predicts **`tip_amount`** from trip-related input features such as trip distance, pickup time, fare amount, borough information, and passenger count.

---

## Project Structure

```text
assignment4/
├── assignment4.ipynb
├── app.py
├── test_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
└── .dockerignore
```

---

## Features Implemented

### FastAPI Endpoints

- **POST `/predict`**  
  Accepts one taxi trip record and returns a predicted tip value.

- **POST `/predict/batch`**  
  Accepts up to 100 trip records and returns predictions for all records.

- **GET `/health`**  
  Returns API health status, model loaded status, and model version.

- **GET `/model/info`**  
  Returns metadata about the loaded model, including:
  - model name
  - version
  - features
  - evaluation metrics
  - training date

### Validation and Error Handling

- Input validation is implemented with **Pydantic**
- Invalid requests return **HTTP 422**
- Unexpected server errors return a structured **HTTP 500** response

### Automated Testing

The API test suite includes coverage for:

- valid single prediction
- valid batch prediction
- missing required fields
- invalid data types
- out-of-range values
- edge case validation
- health endpoint
- model info endpoint

---

## Requirements

### Software
- Python **3.11**
- Docker Desktop or Docker Engine
- Git

### Python Packages
Install dependencies from:

```bash
pip install -r requirements.txt
```

---

## Running the API Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd assignment4
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure model files exist

The API expects these files:

- `models/rfreg_pipe.pkl`
- `models/model_metadata.json`

You can also override the default paths using environment variables:

```bash
export MODEL_PATH=models/rfreg_pipe.pkl
export METADATA_PATH=models/model_metadata.json
```

On Windows PowerShell:

```powershell
$env:MODEL_PATH="models/rfreg_pipe.pkl"
$env:METADATA_PATH="models/model_metadata.json"
```

### 4. Start the FastAPI server

```bash
uvicorn app:app --reload
```

The API should now be available at:

- App: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

---

## Example Requests

### Root Endpoint

```bash
curl http://127.0.0.1:8000/
```

### Single Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 3.5,
    "pickup_hour": 18,
    "pickup_day_of_week": 4,
    "trip_duration_minutes": 15.0,
    "fare_amount": 18.5,
    "pu_borough": "Manhattan",
    "do_borough": "Queens",
    "passenger_count": 1
  }'
```

### Batch Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "trip_distance": 3.5,
        "pickup_hour": 18,
        "pickup_day_of_week": 4,
        "trip_duration_minutes": 15.0,
        "fare_amount": 18.5,
        "pu_borough": "Manhattan",
        "do_borough": "Queens",
        "passenger_count": 1
      },
      {
        "trip_distance": 5.2,
        "pickup_hour": 9,
        "pickup_day_of_week": 1,
        "trip_duration_minutes": 22.0,
        "fare_amount": 24.0,
        "pu_borough": "Brooklyn",
        "do_borough": "Manhattan",
        "passenger_count": 2
      }
    ]
  }'
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Model Info

```bash
curl http://127.0.0.1:8000/model/info
```

---

## Running Tests

Run the automated tests with:

```bash
pytest -v
```

The tests are stored in:

```text
test_app.py
```

---

## Docker Usage

### Build the image

```bash
docker build -t taxi-tip-api .
```

### Run the container

```bash
docker run -p 8000:8000 taxi-tip-api
```

After startup, access the API at:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

---

## Docker Compose

To start services with Docker Compose:

```bash
docker compose up --build
```

To stop services:

```bash
docker compose down
```

If your project includes both an API service and an MLflow tracking service, ensure the API uses the MLflow service name in Compose networking rather than `127.0.0.1`.

---

## MLflow

This assignment includes experiment tracking and model comparison using MLflow. Typical workflow:

1. start the MLflow tracking server
2. log multiple model runs
3. compare metrics such as MAE, RMSE, and R²
4. register the best model
5. load the chosen model for deployment

Example local MLflow UI:

```bash
mlflow ui
```

Default address:

```text
http://127.0.0.1:5000
```

---

## Notes

- The model is loaded **once at startup** and reused across requests.
- Batch predictions are limited to **100 records**.
- Borough values are validated against an allowed set.
- Predictions are rounded to **2 decimal places**.
- Each response includes a unique `prediction_id` for traceability.


