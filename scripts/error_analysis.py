
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Saved predictions from Task 2.5
predictions = pd.read_csv(
    REPORT_DIR / "mobilenet_tuned_test_predictions.csv"
)

# Original test image list
test_df = pd.read_csv("data/splits/test.csv")

# Locate the image path column
possible_columns = [
    "filepath", "file_path", "path",
    "image_path", "filename"
]

path_col = next(
    (col for col in possible_columns
     if col in test_df.columns),
    None
)

if path_col is None:
    raise ValueError(
        f"Could not find image path column. "
        f"Available columns: {list(test_df.columns)}"
    )

assert len(predictions) == len(test_df), (
    "Test split and prediction counts differ."
)

# Match predictions to original images by row
# Requires deterministic, unshuffled test pipeline.
predictions["image_path"] = test_df[path_col].values

# Find incorrect predictions
errors = predictions[
    predictions["true_label"] !=
    predictions["predicted_label"]
].copy()

print("Total test images:", len(predictions))
print("Misclassified images:", len(errors))

# Select up to 10 errors
sample_errors = errors.head(10)

sample_errors.to_csv(
    REPORT_DIR / "error_analysis.csv",
    index=False
)

fig, axes = plt.subplots(
    2, 5, figsize=(16, 8)
)

axes = axes.flatten()

class_names = {
    0: "Real",
    1: "AI-generated"
}

for ax, (_, row) in zip(
    axes, sample_errors.iterrows()
):

    path = Path(row["image_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as img:
        ax.imshow(img.convert("RGB"))

    true_label = class_names[
        int(row["true_label"])
    ]

    predicted_label = class_names[
        int(row["predicted_label"])
    ]

    probability = float(row["ai_probability"])

    ax.set_title(
        f"Actual: {true_label}\n"
        f"Predicted: {predicted_label}\n"
        f"P(AI): {probability:.2f}",
        fontsize=9
    )

    ax.axis("off")

for ax in axes[len(sample_errors):]:
    ax.axis("off")

plt.suptitle(
    "VisionGuard - 10 Misclassified Test Images",
    fontsize=15
)

plt.tight_layout()

output = FIGURE_DIR / "error_analysis_10_images.png"

plt.savefig(
    output,
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("\nError analysis saved!")
print("CSV:", REPORT_DIR / "error_analysis.csv")
print("Figure:", output)
