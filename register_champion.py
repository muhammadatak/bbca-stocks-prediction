import mlflow
from mlflow import MlflowClient
import os
import json
import xgboost as xgb

SQLITE_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(SQLITE_URI)
client = MlflowClient()
MODEL_NAME = "bbca-xgboost-predictor"
MODEL_DIR = "mlruns/1/models/m-d93442ea13c14b289560d0db9a9bc6e6/artifacts"

experiment_name = "stocks_pred_xgboost"
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
    print(f"Created experiment: {experiment_name} (id={experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment: {experiment_name} (id={experiment_id})")

# Load model using xgboost directly
abs_model_dir = os.path.abspath(MODEL_DIR)
model_ubj_path = os.path.join(abs_model_dir, "model.ubj")
print(f"Loading model from: {model_ubj_path}")
model = xgb.XGBClassifier()
model.load_model(model_ubj_path)
print(f"Model loaded: {type(model).__name__}")

with mlflow.start_run(experiment_id=experiment_id, run_name="best-model-import") as run:
    run_id = run.info.run_id
    print(f"Created run: {run_id}")
    mlflow.log_metric("avg_accuracy", 0.5511)
    mlflow.log_metric("avg_f1", 0.4541)
    mlflow.log_metric("avg_roc_auc", 0.5437)
    mlflow.xgboost.log_model(model, artifact_path="model")

print(f"Model logged in run: {run_id}")
mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print(f"Registered model version: {mv.version}")
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Alias champion set ke version {mv.version}")

model_name_q = f"name='{MODEL_NAME}'"
versions = client.search_model_versions(model_name_q)
for v in versions:
    print(f"  Version {v.version}: run_id={v.run_id}, aliases={v.aliases}")

with open("model_registration_result.json", "w") as f:
    json.dump({"run_id": run_id, "version": mv.version, "alias": "champion"}, f, indent=2)
print("Done - saved to model_registration_result.json")
