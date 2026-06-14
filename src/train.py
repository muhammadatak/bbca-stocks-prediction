from pathlib import Path
import os

# ── MLflow config ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING", "true")

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from preprocess import run_split

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("stocks_pred_xgboost")

# ── Load data ────────────────────────────────────────────
clean_data = BASE_DIR / "data" / "processed" / "clean_data.csv"
df = pd.read_csv(clean_data)
X_train, y_train, X_valid, y_valid = run_split(df)

with mlflow.start_run(log_system_metrics=False):
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 9)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=9,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    val_pred = model.predict(X_valid)
    val_prob = model.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, val_pred)
    f1 = f1_score(y_valid, val_pred)

    mlflow.log_metric("avg_accuracy", acc)
    mlflow.log_metric("avg_f1", f1)

    mlflow.xgboost.log_model(model, artifact_path="model")

    print(f"val_accuracy → {acc:.4f}")
    print(f"val_f1       → {f1:.4f}")

print("Training selesai")
