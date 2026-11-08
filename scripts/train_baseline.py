
import sys
from pathlib import Path

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Import the pipeline from Task 2.1
from data_pipeline import build_dataset

# -----------------------------------
# 1. Configuration
# -----------------------------------

SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001

tf.keras.utils.set_random_seed(SEED)

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load training and validation data
# -----------------------------------

train_ds = build_dataset(
    "data/splits/train.csv",
    training=True
)

val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))

# -----------------------------------
# 3. Build baseline CNN
# -----------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),

    # Normalize pixels from 0-255 to 0-1
    tf.keras.layers.Rescaling(1.0 / 255),

    # Convolution block 1
    tf.keras.layers.Conv2D(
        32, (3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution block 2
    tf.keras.layers.Conv2D(
        64, (3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution block 3
    tf.keras.layers.Conv2D(
        128, (3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classification head
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Binary output: probability of AI-generated
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# -----------------------------------
# 4. Compile model
# -----------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# -----------------------------------
# 5. Save model summary
# -----------------------------------

print("\n========== MODEL SUMMARY ==========")

with open("reports/baseline_summary.txt", "w",
          encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

model.summary()

# -----------------------------------
# 6. Training callbacks
# -----------------------------------

checkpoint_path = "models/baseline_best.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# -----------------------------------
# 7. Train
# -----------------------------------

print("\nStarting baseline CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------------
# 8. Save training history
# -----------------------------------

history_df = pd.DataFrame(history.history)
history_df.index += 1
history_df.index.name = "epoch"

history_df.to_csv(
    "reports/baseline_history.csv"
)

# -----------------------------------
# 9. Identify best epoch
# -----------------------------------

best_epoch = int(history_df["val_loss"].idxmin())

print("\nBest epoch:", best_epoch)
print(
    "Best validation loss:",
    history_df.loc[best_epoch, "val_loss"]
)

# -----------------------------------
# 10. Training curves
# -----------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

epochs = history_df.index.to_numpy()

axes[0].plot(
    epochs, history_df["loss"],
    label="Training loss"
)
axes[0].plot(
    epochs, history_df["val_loss"],
    label="Validation loss"
)
axes[0].axvline(
    best_epoch, linestyle="--",
    label=f"Best epoch: {best_epoch}"
)
axes[0].set_title("Baseline CNN - Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy")
axes[0].legend()

axes[1].plot(
    epochs, history_df["accuracy"],
    label="Training accuracy"
)
axes[1].plot(
    epochs, history_df["val_accuracy"],
    label="Validation accuracy"
)
axes[1].axvline(
    best_epoch, linestyle="--",
    label=f"Best epoch: {best_epoch}"
)
axes[1].set_title("Baseline CNN - Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/baseline_training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# -----------------------------------
# 11. Load best checkpoint and evaluate
#     on validation data only
# -----------------------------------

best_model = tf.keras.models.load_model(checkpoint_path)

results = best_model.evaluate(
    val_ds,
    return_dict=True,
    verbose=1
)

print("\nBest checkpoint validation metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nBaseline CNN completed successfully!")
print("Best model:", checkpoint_path)
print("Best epoch:", best_epoch)

