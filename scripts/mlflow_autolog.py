
import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from data_pipeline import build_dataset

SEED = 42
tf.keras.utils.set_random_seed(SEED)

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)

mlflow.set_experiment("VisionGuard_Training")

# Load the best frozen MobileNetV2 checkpoint
model = tf.keras.models.load_model(
    "models/mobilenet_stage1.keras",
    compile=False
)

# Ensure pretrained backbone remains frozen
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

train_ds = build_dataset(
    "data/splits/train.csv",
    training=True
)

val_ds = build_dataset(
    "data/splits/val.csv",
    training=False
)

# Enable MLflow automatic logging
mlflow.tensorflow.autolog(
    log_models=False
)

with mlflow.start_run(
    run_name="MobileNetV2_Autolog"
):
    mlflow.log_param("architecture", "MobileNetV2")
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.set_tag("logging_method", "autolog")
    mlflow.set_tag("seed", "42")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=1
    )

print("\nMLflow autologging completed!")
