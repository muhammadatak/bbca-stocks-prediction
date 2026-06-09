from pathlib import Path

import os

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from preprocess import run_split

BASE_DIR = Path(__file__).resolve().parent.parent
clean_data = BASE_DIR / "data" / "processed" / "clean_data.csv"


df = pd.read_csv(clean_data)

X_train, y_train, X_valid, y_valid = run_split(df)

mlflow.set_experiment("stocks_pred_xgboost")

with mlflow.start_run():
    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cpu"
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    val_pred = model.predict(X_valid)
    val_prob = model.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, val_pred)
    f1 = f1_score(y_valid, val_pred)
    auc = roc_auc_score(y_valid, val_prob)

    mlflow.log_metric("avg_accuracy", np.mean(acc))
    mlflow.log_metric("avg_f1", np.mean(f1))
    mlflow.log_metric("avg_roc_auc", np.mean(auc))

    mlflow.xgboost.log_model(model, "model")

    print(f"val_accuracy → {acc:.4f}")
    print(f"val_f1       → {f1:.4f}")
    print(f"val_roc_auc  → {auc:.4f}")

print("Training selesai")
