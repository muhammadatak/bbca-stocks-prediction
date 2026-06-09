import json
import os
import sys

import mlflow
import yaml
from mlflow import MlflowClient

with open("thresholds.yaml") as f:
    thresholds = yaml.safe_load(f)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()
MODEL_NAME = os.getenv("MODEL_NAME", "bbca-xgboost-predictor")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stocks_pred_xgboost")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "challenger")

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["metrics.avg_accuracy DESC"],
    max_results=1,
)

if runs.empty:
    sys.exit(1)

best_run = runs.iloc[0]
run_id = best_run["run_id"]
avg_accuracy = best_run.get("metrics.avg_accuracy")
avg_f1 = best_run.get("metrics.avg_f1")
avg_roc_auc = best_run.get("metrics.avg_roc_auc")

if avg_accuracy is None or avg_f1 is None or avg_roc_auc is None:
    sys.exit(1)

print(f"avg_accuracy : {avg_accuracy:.4f}")
print(f"avg_f1       : {avg_f1:.4f}")
print(f"avg_roc_auc  : {avg_roc_auc:.4f}")

passed = (
    avg_accuracy >= thresholds["avg_accuracy_min"]
    and avg_f1 >= thresholds["avg_f1_min"]
    and avg_roc_auc >= thresholds["avg_roc_auc_min"]
)

if not passed:
    sys.exit(1)


mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, mv.version)


with open("model_registration_result.json", "w") as f:
    json.dump(
        {
            "run_id": run_id,
            "version": mv.version,
            "alias": MODEL_ALIAS,
            "metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_f1": avg_f1,
                "avg_roc_auc": avg_roc_auc,
            },
        },
        f,
        indent=2,
    )
