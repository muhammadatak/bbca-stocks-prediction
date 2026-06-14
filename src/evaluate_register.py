
import json
import os
import sys
from pathlib import Path

# ── MLflow config ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "bbca-xgboost-predictor"
EXPERIMENT_NAME = "stocks_pred_xgboost"

client = MlflowClient()

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["start_time DESC"],
    max_results=1
)

latest_run = runs.iloc[0]
run_id = latest_run["run_id"]
run = client.get_run(run_id)

metrics = run.data.metrics

avg_accuracy = metrics.get("avg_accuracy")
avg_f1 = metrics.get("avg_f1")

thresholds = {
    "avg_accuracy": 0.30,
    "avg_f1": 0.30,
}

passed = (
    avg_accuracy >= thresholds["avg_accuracy"] and
    avg_f1 >= thresholds["avg_f1"]  )

print("accuracy:", avg_accuracy)
print("f1:", avg_f1)
print("passed:", passed)


# register
if passed:

    model_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(
        model_uri,
        MODEL_NAME
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "challenger",
        model_version.version
    )

    print("✅ Registered as CHALLENGER (version {})".format(model_version.version))

    # --- Bandingkan dengan champion ---
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        # Belum ada champion → langsung promosikan
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)
        print("🏆 Promoted to CHAMPION (no previous champion)")
    else:
        # Ambil metrik champion dari run asalnya
        champion_run_id = champion.run_id
        champion_run = client.get_run(champion_run_id)
        champion_metrics = champion_run.data.metrics

        champ_acc = champion_metrics.get("avg_accuracy", 0)
        champ_f1 = champion_metrics.get("avg_f1", 0)

        print("\n--- Champion vs Challenger ---")
        print(f"Champion  → acc: {champ_acc:.4f}, f1: {champ_f1:.4f}")
        print(f"Challenger → acc: {avg_accuracy:.4f}, f1: {avg_f1:.4f}")

        challenger_wins = (
            avg_accuracy >= champ_acc and
            avg_f1 >= champ_f1)

        if challenger_wins:
            client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)
            print("🏆 Challenger promoted to CHAMPION!")
        else:
            print("❌ Challenger did not beat the champion")

else:
    print("❌ Rejected")

# ── Push metrics ke Pushgateway (Scenario A) ──────────
try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    registry = CollectorRegistry()
    Gauge("model_accuracy", "Latest model accuracy", registry=registry).set(avg_accuracy)
    Gauge("model_f1", "Latest model F1", registry=registry).set(avg_f1)
    push_to_gateway("localhost:9091", job="model-evaluator", registry=registry)
    print("✅ Metrics pushed to Pushgateway")
except Exception as e:
    print(f"⚠️  Pushgateway not available: {e}")