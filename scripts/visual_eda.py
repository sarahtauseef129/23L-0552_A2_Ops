
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

# Paths
AUDIT_FILE = Path("reports/dataset_audit.csv")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Read audit results
df = pd.read_csv(AUDIT_FILE)

# Remove corrupt images from visualization
df = df[df["corrupt"] == False].copy()

random.seed(42)

# --------------------------------------
# 1. Sample image grid
# --------------------------------------

classes = ["AI-generated", "Real"]

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for row, label in enumerate(classes):
    samples = df[df["class"] == label].sample(
        n=5, random_state=42
    )

    for col, (_, record) in enumerate(samples.iterrows()):
        image_path = record["filepath"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                label, fontsize=12
            )

    axes[row, 0].text(
        -0.1, 0.5, label,
        transform=axes[row, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=12
    )

plt.suptitle("Sample Images: AI-generated vs Real")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "sample_images.png",
    dpi=300, bbox_inches="tight"
)
plt.show()


# --------------------------------------
# 2. Histograms
# --------------------------------------

features = {
    "width": "Image Width (pixels)",
    "height": "Image Height (pixels)",
    "file_size_kb": "File Size (KB)"
}

for feature, xlabel in features.items():

    plt.figure(figsize=(9, 5))

    for label in classes:
        subset = df[df["class"] == label]

        plt.hist(
            subset[feature],
            bins=30,
            alpha=0.5,
            label=label
        )

    plt.xlabel(xlabel)
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of {xlabel}")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.savefig(
        FIGURE_DIR / f"{feature}_histogram.png",
        dpi=300, bbox_inches="tight"
    )

    plt.show()

print("EDA completed!")
print("Figures saved inside reports/figures/")
