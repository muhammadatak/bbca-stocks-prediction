import json
import os
import sys

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# ── Config ──────────────────────────────────────────────
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "thresholds.yaml")
with open(THRESHOLDS_PATH) as f:
    thresholds = yaml.safe_load(f)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()
MODEL_NAME = os.getenv("MODEL_NAME", "bbca-xgboost-predictor")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stocks_pred_xgboost")
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"

# ── 1. Cari run terbaik (challenger) ───────────────────
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["metrics.avg_accuracy DESC"],
    max_results=1,
)

if runs.empty:
    print("❌ Tidak ada run ditemukan.")
    sys.exit(1)

best_run = runs.iloc[0]
run_id = best_run["run_id"]
challenger_metrics = {
    "avg_accuracy": best_run.get("metrics.avg_accuracy"),
    "avg_f1": best_run.get("metrics.avg_f1"),
    "avg_roc_auc": best_run.get("metrics.avg_roc_auc"),
}

if any(v is None for v in challenger_metrics.values()):
    print("❌ Metrik tidak lengkap.")
    sys.exit(1)

print("=" * 50)
print("  CHALLENGER (run terbaik)")
print("=" * 50)
print(f"  run_id      : {run_id}")
print(f"  avg_accuracy: {challenger_metrics['avg_accuracy']:.4f}")
print(f"  avg_f1      : {challenger_metrics['avg_f1']:.4f}")
print(f"  avg_roc_auc : {challenger_metrics['avg_roc_auc']:.4f}")

# ── 2. Cek threshold ────────────────────────────────────
passed = (
    challenger_metrics["avg_accuracy"] >= thresholds["avg_accuracy_min"]
    and challenger_metrics["avg_f1"] >= thresholds["avg_f1_min"]
    and challenger_metrics["avg_roc_auc"] >= thresholds["avg_roc_auc_min"]
)

if not passed:
    print("\n❌ Challenger TIDAK lolos threshold. Stop.")
    sys.exit(1)
print("\n✅ Challenger lolos threshold.")

# ── 3. Register challenger ──────────────────────────────
mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)
print(f"📦 Registered: {MODEL_NAME} v{mv.version} (alias: {CHALLENGER_ALIAS})")

# ── 4. Bandingkan dengan champion ──────────────────────
promoted = False
champion_info = None

try:
    champion_mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    champion_run = mlflow.get_run(champion_mv.run_id)
    champion_metrics = champion_run.data.metrics

    print("\n" + "=" * 50)
    print("  CHAMPION (existing)")
    print("=" * 50)
    print(f"  version     : v{champion_mv.version}")
    print(f"  run_id      : {champion_mv.run_id}")
    print(f"  avg_accuracy: {champion_metrics.get('avg_accuracy', 0):.4f}")
    print(f"  avg_f1      : {champion_metrics.get('avg_f1', 0):.4f}")
    print(f"  avg_roc_auc : {champion_metrics.get('avg_roc_auc', 0):.4f}")

    # Gunakan accuracy sebagai metrik utama perbandingan
    champ_acc = champion_metrics.get("avg_accuracy", 0)
    chall_acc = challenger_metrics["avg_accuracy"]

    print(
        f"\n📊 Perbandingan accuracy: challenger={chall_acc:.4f} vs champion={champ_acc:.4f}"
    )

    if chall_acc > champ_acc:
        print("🏆 Challenger LEBIH BAIK dari champion!")
        promoted = True
    else:
        print("⏸️  Champion tetap lebih baik, tidak ada promosi.")

except MlflowException:
    print("\n⚠️  Belum ada champion. Challenger langsung promosi jadi champion.")
    promoted = True

# ── 5. Promosi ke champion jika layak ───────────────────
if promoted:
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
    print(f"🚀 Promosi: {MODEL_NAME} v{mv.version} → alias '{CHAMPION_ALIAS}'")

# ── 6. Simpan hasil ─────────────────────────────────────
result = {
    "challenger": {
        "run_id": run_id,
        "version": mv.version,
        "alias": CHALLENGER_ALIAS,
        "metrics": challenger_metrics,
    },
    "promoted_to_champion": promoted,
}

if champion_info:
    result["champion_previous"] = champion_info

with open("model_registration_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n" + "=" * 50)
if promoted:
    print("  ✅ FINAL: Challenger → CHAMPION!")
else:
    print("  ⏸️  FINAL: Champion tetap dipertahankan.")
print(f"  Model: {MODEL_NAME} v{mv.version}")
print("=" * 50)
