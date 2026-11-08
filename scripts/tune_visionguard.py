
from pathlib import Path
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from data_pipeline import build_dataset

SEED = 42
tf.keras.utils.set_random_seed(SEED)

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ======================================
# 1. Load training and validation data
# ======================================

train_ds = build_dataset(
    "data/splits/train.csv",
    training=True
)

val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

# ======================================
# 2. Define the tunable MobileNetV2
# ======================================

def build_model(hp):

    # Start each trial from the same
    # trained Stage 1 checkpoint
    model = tf.keras.models.load_model(
        "models/mobilenet_stage1.keras",
        compile=False
    )

    # Hyperparameter 1: learning rate
    lr = hp.Choice(
        "learning_rate",
        [1e-4, 1e-5]
    )

    # Hyperparameter 2: dropout rate
    dropout = hp.Choice(
        "dropout",
        [0.2, 0.3, 0.5]
    )

    # Hyperparameter 3: unfrozen layers
    unfrozen = hp.Choice(
        "unfrozen_layers",
        [10, 20, 30]
    )

    # Find the nested MobileNetV2 model
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenet" in layer.name.lower()
    )

    backbone.trainable = True

    # Freeze earlier backbone layers
    for layer in backbone.layers[:-unfrozen]:
        layer.trainable = False

    # Unfreeze only the selected final layers
    # but keep BatchNormalization frozen
    for layer in backbone.layers[-unfrozen:]:
        layer.trainable = not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        )

    # Update dropout in classification head
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout

    # Compile after changing trainability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


# ======================================
# 3. Configure Keras Tuner
# ======================================

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective(
        "val_loss",
        direction="min"
    ),
    max_trials=8,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="visionguard_mobilenet",
    overwrite=True,
    seed=SEED
)

# ======================================
# 4. Run hyperparameter search
# ======================================

print("\nStarting VisionGuard tuning...")

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ],
    verbose=2
)

# ======================================
# 5. Display results
# ======================================

print("\n========== TUNER RESULTS ==========")
tuner.results_summary(num_trials=8)

rows = []

for trial in tuner.oracle.trials.values():
    hp = trial.hyperparameters

    rows.append({
        "trial": trial.trial_id,
        "learning_rate": hp.get("learning_rate"),
        "dropout": hp.get("dropout"),
        "unfrozen_layers": hp.get("unfrozen_layers"),
        "val_loss": trial.score,
        "status": trial.status
    })

results_df = pd.DataFrame(rows)

results_df = results_df.sort_values(
    "val_loss",
    na_position="last"
)

results_df.to_csv(
    "reports/visionguard_tuner_results.csv",
    index=False
)

print("\n========== TRIAL COMPARISON ==========")
print(results_df.to_string(index=False))

# ======================================
# 6. Best hyperparameters
# ======================================

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\n========== BEST SETTINGS ==========")
print("Learning rate:", best_hp.get("learning_rate"))
print("Dropout:", best_hp.get("dropout"))
print("Unfrozen layers:", best_hp.get("unfrozen_layers"))

# ======================================
# 7. Save best tuned model separately
# ======================================

best_model = tuner.get_best_models(1)[0]

best_model.save(
    "models/mobilenet_tuned_best.keras"
)

print("\nTuning completed successfully!")
print("Saved: reports/visionguard_tuner_results.csv")
print("Saved: models/mobilenet_tuned_best.keras")
