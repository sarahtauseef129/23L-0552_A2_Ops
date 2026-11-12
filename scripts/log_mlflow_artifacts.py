
import os
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from data_pipeline import build_dataset

# Configuration
SEED = 42
MODEL_PATH = Path("models/mobilenet_best.keras")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# Connect to DagsHub
mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)

mlflow.set_experiment("VisionGuard_Training")

# Load previously trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Validation data only (NOT test data)
val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

# Get actual labels
y_true = np.concatenate([
    labels.numpy()
    for _, labels in val_ds
]).astype(int)

# Predict probability of AI-generated class
y_prob = model.predict(val_ds, verbose=1).ravel()

# Temporary default threshold
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# ------------------------------
# 1. Confusion matrix
# ------------------------------

cm = confusion_matrix(
    y_true, y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "AI-generated"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("MobileNetV2 Validation Confusion Matrix")
plt.tight_layout()

cm_path = FIGURE_DIR / "validation_confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# 2. ROC curve
# ------------------------------

fpr, tpr, thresholds = roc_curve(
    y_true, y_prob
)

auc_score = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MobileNetV2 Validation ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = FIGURE_DIR / "validation_roc_curve.png"
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# 3. Log artifacts
# ------------------------------

with mlflow.start_run(
    run_name="MobileNetV2_Validation_Artifacts"
):

    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("dataset_split", "validation")

    mlflow.log_metric("validation_roc_auc", auc_score)

    mlflow.set_tag("seed", str(SEED))
    mlflow.set_tag("evaluation_type", "validation_only")

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()

    mlflow.set_tag("git_commit", git_commit)

    # DVC pointer identifies the dataset version
    if Path("data/raw.dvc").exists():
        mlflow.log_artifact(
            "data/raw.dvc",
            artifact_path="data_version"
        )

    # Log figures
    mlflow.log_artifact(str(cm_path), artifact_path="figures")
    mlflow.log_artifact(str(roc_path), artifact_path="figures")

    # Log split lists
    for name in ["train", "val", "test"]:
        mlflow.log_artifact(
            f"data/splits/{name}.csv",
            artifact_path="splits"
        )

    # Log package list
    mlflow.log_artifact("requirements.txt")

    # Log model file
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="model"
    )

print("\nAll MLflow artifacts uploaded successfully!")
print(f"Validation ROC-AUC: {auc_score:.4f}")
