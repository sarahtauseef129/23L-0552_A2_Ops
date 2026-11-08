
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import build_dataset

# =====================================
# 1. CONFIGURATION
# =====================================

SEED = 42
IMG_SIZE = (224, 224)

tf.keras.utils.set_random_seed(SEED)

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# =====================================
# 2. LOAD DATASETS
# =====================================

train_ds = build_dataset(
    "data/splits/train.csv",
    training=True
)

val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

# =====================================
# 3. LOAD PRETRAINED MOBILENETV2
# =====================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze all pretrained layers
base_model.trainable = False

# =====================================
# 4. BUILD CLASSIFICATION HEAD
# =====================================

inputs = tf.keras.Input(shape=(224, 224, 3))

# Convert pixels from [0,255] to [-1,1]
x = tf.keras.layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(inputs)

# Keep BatchNorm in inference mode
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs, outputs,
    name="VisionGuard_MobileNetV2"
)

# =====================================
# 5. HELPER FUNCTIONS
# =====================================

def compile_model(lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )


def create_callbacks(checkpoint):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


def save_trainable_layers(stage):
    path = f"reports/mobilenet_{stage}_layers.txt"

    with open(path, "w", encoding="utf-8") as f:
        for layer in base_model.layers:
            f.write(
                f"{layer.name}: "
                f"trainable={layer.trainable}\n"
            )

    print(f"Trainable layers saved: {path}")


# =====================================
# 6. STAGE 1: FEATURE EXTRACTION
# =====================================

print("\n========== STAGE 1 ==========")
print("Training classification head...")

compile_model(lr=0.001)

model.summary()

save_trainable_layers("stage1")

stage1_path = "models/mobilenet_stage1.keras"

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=create_callbacks(stage1_path)
)

# Record stage 1 history
h1 = pd.DataFrame(history1.history)
h1["stage"] = "feature_extraction"
h1["epoch"] = range(1, len(h1) + 1)

h1.to_csv(
    "reports/mobilenet_stage1_history.csv",
    index=False
)

# Restore the best stage 1 weights
model.load_weights(stage1_path)

print("\nStage 1 complete!")

# =====================================
# 7. STAGE 2: FINE-TUNING
# =====================================

print("\n========== STAGE 2 ==========")
print("Fine-tuning last MobileNetV2 layers...")

base_model.trainable = True

# Freeze all except the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

# Recompile after modifying trainable layers
compile_model(lr=0.00001)

save_trainable_layers("stage2")

print("\nFine-tuning model summary:")
model.summary()

stage2_path = "models/mobilenet_stage2.keras"

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=create_callbacks(stage2_path)
)

h2 = pd.DataFrame(history2.history)
h2["stage"] = "fine_tuning"
h2["epoch"] = range(1, len(h2) + 1)

h2.to_csv(
    "reports/mobilenet_stage2_history.csv",
    index=False
)

# =====================================
# 8. COMPARE BOTH STAGES
# =====================================

best1 = h1["val_loss"].min()
best2 = h2["val_loss"].min()

print("\n========== COMPARISON ==========")

print("Stage 1 best validation loss:", best1)
print("Stage 2 best validation loss:", best2)

if best2 < best1:
    best_path = stage2_path
    best_stage = "Stage 2 - Fine-tuning"
    best_epoch = int(h2["val_loss"].idxmin() + 1)
else:
    best_path = stage1_path
    best_stage = "Stage 1 - Feature extraction"
    best_epoch = int(h1["val_loss"].idxmin() + 1)

# Load best performing stage
best_model = tf.keras.models.load_model(best_path)

best_model.save("models/mobilenet_best.keras")

print("\nSelected:", best_stage)
print("Best epoch:", best_epoch)

# =====================================
# 9. PLOT TRAINING CURVES
# =====================================

combined = pd.concat([h1, h2], ignore_index=True)
epochs = np.arange(1, len(combined) + 1)

stage_boundary = len(h1) + 0.5

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    epochs, combined["loss"],
    label="Training loss"
)

axes[0].plot(
    epochs, combined["val_loss"],
    label="Validation loss"
)

axes[0].axvline(
    stage_boundary,
    color="gray",
    linestyle="--",
    label="Fine-tuning begins"
)

axes[0].set_title("MobileNetV2 Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(
    epochs, combined["accuracy"],
    label="Training accuracy"
)

axes[1].plot(
    epochs, combined["val_accuracy"],
    label="Validation accuracy"
)

axes[1].axvline(
    stage_boundary,
    color="gray",
    linestyle="--",
    label="Fine-tuning begins"
)

axes[1].set_title("MobileNetV2 Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/mobilenet_training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# 10. VALIDATION EVALUATION
# =====================================

print("\n========== FINAL VALIDATION ==========")

results = best_model.evaluate(
    val_ds,
    return_dict=True
)

for key, value in results.items():
    print(f"{key}: {value:.4f}")

print("\nMobileNetV2 training completed!")
print("Best model: models/mobilenet_best.keras")
