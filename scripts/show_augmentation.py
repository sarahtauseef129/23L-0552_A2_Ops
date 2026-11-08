
import matplotlib.pyplot as plt
from pathlib import Path
from data_pipeline import build_dataset

train_ds = build_dataset(
    "data/splits/train.csv",
    training=True
)

images, labels = next(iter(train_ds))

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    image = images[i].numpy()

    ax.imshow(image.clip(0, 255).astype("uint8"))
    ax.set_title(
        "AI-generated" if labels[i].numpy() == 1
        else "Real"
    )
    ax.axis("off")

plt.suptitle("Augmented Training Batch")
plt.tight_layout()

output = Path("reports/figures")
output.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output / "augmented_batch.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Augmented batch figure saved!")
