
import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from mlflow import MlflowClient
from pathlib import Path

# Connect to DagsHub
mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)
mlflow.set_experiment("VisionGuard_Registry")

MODEL_NAME = "VisionGuard_Classifier"

models_to_register = [
    (
        "MobileNetV2_FineTuned",
        "models/mobilenet_best.keras",
        0.437259
    ),
    (
        "MobileNetV2_Tuned",
        "models/mobilenet_tuned_best.keras",
        0.418920
    )
]

registered_versions = []

for run_name, model_path, val_loss in models_to_register:

    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    print(f"\nRegistering {run_name}...")

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("architecture", "MobileNetV2")
        mlflow.log_param("source_checkpoint", model_path)
        mlflow.log_metric("historical_best_val_loss", val_loss)

        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME
        )

        version = model_info.registered_model_version

        registered_versions.append(
            (run_name, version)
        )

        print(f"Registered version: {version}")

# Assign champion alias to tuned model
client = MlflowClient()

tuned_version = registered_versions[-1][1]

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=str(tuned_version)
)

print("\n========== REGISTRY RESULTS ==========")

for name, version in registered_versions:
    print(f"{name}: version {version}")

print(f"\nChampion: version {tuned_version}")
print("Model registry completed!")
