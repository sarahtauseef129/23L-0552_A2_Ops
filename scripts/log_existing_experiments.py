
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)
mlflow.set_experiment("VisionGuard_Training")

def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()

def log_existing_run(name, history_file, model_file, params):
    df = pd.read_csv(history_file)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "seed": "42",
            "logging_method": "manual",
            "dataset": "VisionGuard"
        })

        for i, row in df.iterrows():
            step = int(row["epoch"]) if "epoch" in df else i + 1

            for metric in [
                "loss", "accuracy", "auc",
                "precision", "recall",
                "val_loss", "val_accuracy",
                "val_auc", "val_precision",
                "val_recall"
            ]:
                if metric in df.columns:
                    mlflow.log_metric(
                        metric,
                        float(row[metric]),
                        step=step
                    )

        mlflow.log_artifact(history_file)

        if Path(model_file).exists():
            mlflow.log_artifact(
                model_file,
                artifact_path="models"
            )

        print("Logged:", name)

log_existing_run(
    "Baseline_CNN",
    "reports/baseline_history.csv",
    "models/baseline_best.keras",
    {"architecture": "Custom CNN", "learning_rate": 0.001}
)

log_existing_run(
    "MobileNetV2_Stage1",
    "reports/mobilenet_stage1_history.csv",
    "models/mobilenet_stage1.keras",
    {"architecture": "MobileNetV2", "stage": "frozen"}
)

log_existing_run(
    "MobileNetV2_Stage2",
    "reports/mobilenet_stage2_history.csv",
    "models/mobilenet_stage2.keras",
    {"architecture": "MobileNetV2", "stage": "fine_tuning",
     "learning_rate": 0.00001}
)

print("\nAll existing training runs logged to DagsHub!")
