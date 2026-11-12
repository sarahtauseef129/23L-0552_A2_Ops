
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from data_pipeline import build_dataset

tf.keras.utils.set_random_seed(42)

Path("reports").mkdir(exist_ok=True)

models = {
    "baseline": "models/baseline_best.keras",
    "mobilenet_tuned": "models/mobilenet_tuned_best.keras"
}

val_df = pd.read_csv("data/splits/val.csv")

# Validation dataset is not shuffled
val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

y_true = val_df["label"].to_numpy().astype(int)

results = []

for name, model_path in models.items():

    print(f"\nEvaluating thresholds for {name}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    probabilities = model.predict(
        val_ds, verbose=0
    ).ravel()

    for threshold in np.arange(
        0.20, 0.81, 0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        # False accusation costs twice as much
        cost = 2 * fp + fn

        results.append({
            "model": name,
            "threshold": round(float(threshold), 2),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "cost": int(cost),
            "precision": precision_score(
                y_true, predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions,
                zero_division=0
            )
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

best_rows = []

for name in models:

    subset = results_df[
        results_df["model"] == name
    ]

    # Lowest cost, highest F1 for ties
    best = subset.sort_values(
        ["cost", "f1"],
        ascending=[True, False]
    ).iloc[0]

    best_rows.append(best)

best_df = pd.DataFrame(best_rows)

best_df.to_csv(
    "reports/selected_thresholds.csv",
    index=False
)

print("\n========== SELECTED THRESHOLDS ==========")

print(
    best_df[
        [
            "model", "threshold",
            "false_positives",
            "false_negatives",
            "cost", "precision",
            "recall", "f1"
        ]
    ].to_string(index=False)
)

print("\nThreshold selection completed!")
print("No test images were evaluated.")
