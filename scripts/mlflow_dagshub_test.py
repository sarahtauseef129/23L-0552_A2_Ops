
import mlflow
import os

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)

mlflow.set_experiment("VisionGuard_DagsHub_Test")

with mlflow.start_run(run_name="dagshub_connection_test"):
    mlflow.log_param("model", "connection_test")
    mlflow.log_param("seed", 42)
    mlflow.log_metric("test_metric", 0.95)
    mlflow.set_tag("purpose", "Verify DagsHub MLflow tracking")

print("DagsHub MLflow test run logged successfully!")
