import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import yfinance as yf
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import socket
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

from src.preprocess import add_features




app = FastAPI()

Instrumentator().instrument(app).expose(app)

prediction_probability = Histogram(
    "prediction_probability",
    "Distribution of prediction confidence"
)

prediction_class_total = Counter(
    "prediction_class_total",
    "Predictions by class",
    ["signal"]
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "bbca-xgboost-predictor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_STAGE = os.getenv("MODEL_STAGE")
TICKER = os.getenv("TICKER", "BBCA.JK")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



def load_registry_model():
    client = MlflowClient()
    candidate_uris = []

    if MODEL_ALIAS:
        candidate_uris.append(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    if MODEL_STAGE:
        candidate_uris.append(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest_version = max(versions, key=lambda mv: int(mv.version))
        candidate_uris.append(f"models:/{MODEL_NAME}/{latest_version.version}")

    last_error = None
    for uri in candidate_uris:
        try:
            return uri, mlflow.xgboost.load_model(uri)
        except (MlflowException, OSError) as err:
            last_error = err

    raise RuntimeError(
        f"Tidak dapat memuat model registry '{MODEL_NAME}'. "
        "Set MODEL_ALIAS/MODEL_STAGE yang valid atau register model terlebih dahulu."
    ) from last_error


MODEL_URI, model = load_registry_model()

FEATURE_COLUMNS = [
    "Close",
    "RSI",
    "EMA_9",
    "SMA_5",
    "SMA_10",
    "SMA_15",
    "SMA_30",
    "MACD",
    "MACD_signal",
]



class PredictResponse(BaseModel):
    prediction: int
    probability_up: float
    signal: str


@app.get("/")
def root():
    return {
        "message": "Hello",
        "container": socket.gethostname()
    } 


@app.post("/predict", response_model=PredictResponse)
def predict():
    df = yf.download(TICKER, period="50d", auto_adjust=True, progress=False)

    if df.empty:
        raise HTTPException(
            status_code=422, detail="Gagal mengambil data harga terbaru."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    fe_df = add_features(df)
    fe_df = fe_df.dropna(subset=FEATURE_COLUMNS)
    if fe_df.empty:
        raise HTTPException(
            status_code=422,
            detail="Data tidak cukup untuk preprocessing.",
        )

    latest_df = fe_df.iloc[[-1]][FEATURE_COLUMNS]
    prediction = int(model.predict(latest_df)[0])
    probability = float(model.predict_proba(latest_df)[0][1])

    signal = "BUY" if prediction == 1 else "SELL"

    prediction_probability.observe(probability)

    prediction_class_total.labels(
        signal=signal
    ).inc()
    return PredictResponse(
        prediction=prediction,
        probability_up=round(probability, 4),
        signal=signal,
    )
