
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

for stage in ["stage1", "stage2"]:
    df = pd.read_csv(
        f"reports/mobilenet_{stage}_history.csv"
    )

    best_epoch = int(
        df.loc[df["val_loss"].idxmin(), "epoch"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(
        axes, ["loss", "accuracy"]
    ):
        ax.plot(
            df["epoch"], df[metric],
            label="Training"
        )
        ax.plot(
            df["epoch"], df[f"val_{metric}"],
            label="Validation"
        )
        ax.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best epoch: {best_epoch}"
        )
        ax.set_title(f"{stage.upper()} - {metric.title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"reports/figures/mobilenet_{stage}_curves.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"{stage}: best epoch = {best_epoch}")

print("Both stage plots saved!")
