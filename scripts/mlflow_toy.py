
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

tf.keras.utils.set_random_seed(42)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Use a local MLflow tracking directory
mlflow.set_tracking_uri(
    Path("mlruns").resolve().as_uri()
)

mlflow.set_experiment("VisionGuard_MLflow_Toy")

# Small built-in dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

x_train = x_train[:5000].astype("float32") / 255
y_train = y_train[:5000]

x_val = x_test[:1000].astype("float32") / 255
y_val = y_test[:1000]

# Train two runs with different learning rates
for lr in [0.001, 0.0001]:

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    with mlflow.start_run(
        run_name=f"learning_rate_{lr}"
    ):

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        mlflow.set_tag("task", "MLflow toy lab")

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                64, activation="relu"
            ),
            tf.keras.layers.Dense(
                10, activation="softmax"
            )
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=64,
            verbose=1
        )

        # Log metrics for every epoch
        for epoch in range(
            len(history.history["loss"])
        ):
            for metric, values in history.history.items():
                mlflow.log_metric(
                    metric,
                    float(values[epoch]),
                    step=epoch + 1
                )

        # Create and log accuracy plot
        plt.figure(figsize=(8, 5))
        plt.plot(
            history.history["accuracy"],
            label="Train"
        )
        plt.plot(
            history.history["val_accuracy"],
            label="Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Rate = {lr}")
        plt.legend()

        plot_path = (
            f"reports/figures/mlflow_toy_{lr}.png"
        )
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        # Save and log model
        model_path = f"models/mlflow_toy_{lr}.keras"
        Path("models").mkdir(exist_ok=True)

        model.save(model_path)
        mlflow.log_artifact(
            model_path,
            artifact_path="model"
        )

        print(
            f"\nCompleted MLflow run with LR={lr}"
        )

print("\nBoth MLflow experiments completed!")
