import mlflow
from mlflow import MlflowClient

client = MlflowClient()

try:
    versions = client.search_model_versions("name='bbca-xgboost-predictor'")
    print(f"Model versions found: {len(versions)}")
    for v in versions:
        print(f"  Version {v.version}: run_id={v.run_id}, aliases={v.aliases}, stage={v.current_stage}")
except Exception as e:
    print(f"Error: {e}")

# Also check experiment runs
experiment = mlflow.get_experiment_by_name("stocks_pred_xgboost")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"\nRuns found: {len(runs)}")
    if not runs.empty:
        print(runs[["run_id", "metrics.avg_accuracy", "metrics.avg_f1", "metrics.avg_roc_auc"]])
else:
    print("Experiment 'stocks_pred_xgboost' not found")
