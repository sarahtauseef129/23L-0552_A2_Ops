
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from data_pipeline import build_dataset

tf.keras.utils.set_random_seed(42)

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Previously selected validation thresholds
threshold_df = pd.read_csv(
    "reports/selected_thresholds.csv"
)

thresholds = dict(
    zip(
        threshold_df["model"],
        threshold_df["threshold"]
    )
)

models = {
    "baseline": "models/baseline_best.keras",
    "mobilenet_tuned": "models/mobilenet_tuned_best.keras"
}

# Load untouched test dataset
test_ds = build_dataset(
    "data/splits/test.csv",
    training=False
)

# Collect labels directly from the dataset to
# preserve exactly the same order as predictions.
y_true = np.concatenate([
    np.asarray(labels).reshape(-1)
    for _, labels in test_ds
]).astype(int)

assert set(np.unique(y_true)).issubset({0, 1})

results = []

for name, model_path in models.items():

    print(f"\n========== {name.upper()} ==========")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    threshold = float(thresholds[name])

    # The model returns probability of class 1 (AI).
    y_prob = model.predict(
        test_ds,
        verbose=1
    ).reshape(-1)

    assert len(y_prob) == len(y_true), (
        "Prediction and label counts do not match."
    )

    assert np.all(
        (y_prob >= 0) & (y_prob <= 1)
    ), "Expected probabilities between 0 and 1."

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_true, y_prob)

    results.append({
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Real", "AI-generated"],
            zero_division=0
        )
    )

    # Save predictions for later error analysis.
    pd.DataFrame({
        "index": np.arange(len(y_true)),
        "true_label": y_true,
        "predicted_label": y_pred,
        "ai_probability": y_prob
    }).to_csv(
        REPORT_DIR / f"{name}_test_predictions.csv",
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "AI-generated"]
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{name} - Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{name}_test_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_true, y_prob
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr, tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - Test ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{name}_test_roc_curve.png",
        dpi=300
    )

    plt.close()

# Save model comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "final_model_comparison.csv",
    index=False
)

print("\n========== FINAL TEST COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFinal test evaluation completed!")
