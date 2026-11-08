
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
from pathlib import Path

tf.keras.utils.set_random_seed(42)

Path("reports").mkdir(exist_ok=True)

# Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Use only training data for tuning
x_train = x_train.astype("float32") / 255.0

# Separate validation data
x_val = x_train[-5000:]
y_val = y_train[-5000:]

x_train = x_train[:-5000]
y_train = y_train[:-5000]

# Define tunable model
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            units=hp.Choice(
                "units", [32, 64, 128]
            ),
            activation="relu"
        ),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    learning_rate = hp.Choice(
        "learning_rate",
        [0.01, 0.001, 0.0001]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Random search
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=6,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="fashion_mnist_toy",
    overwrite=True,
    seed=42
)

tuner.search(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2
        )
    ]
)

# Display results
print("\n========== TUNER RESULTS ==========")
tuner.results_summary()

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBEST HYPERPARAMETERS")
print("Units:", best_hp.get("units"))
print("Learning rate:", best_hp.get("learning_rate"))

# Save all trial results
rows = []

for trial in tuner.oracle.trials.values():
    rows.append({
        "trial_id": trial.trial_id,
        "units": trial.hyperparameters.get("units"),
        "learning_rate": trial.hyperparameters.get(
            "learning_rate"
        ),
        "validation_accuracy": trial.score,
        "status": trial.status
    })

pd.DataFrame(rows).to_csv(
    "reports/tuner_toy_results.csv",
    index=False
)

print("\nResults saved to reports/tuner_toy_results.csv")
