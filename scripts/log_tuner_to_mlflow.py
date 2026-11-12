
import os
import subprocess
from pathlib import Path

import pandas as pd
import mlflow

# Connect to DagsHub
mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)

mlflow.set_experiment("VisionGuard_Tuning_Comparison")

# Read actual tuner results
csv_path = Path("reports/visionguard_tuner_results.csv")
df = pd.read_csv(csv_path)

# Six trials arranged so consecutive
# trials change one hyperparameter
selected_trials = [0, 2, 6, 4, 5, 1]

df["trial"] = df["trial"].astype(int)

selected = df.set_index("trial").loc[
    selected_trials
].reset_index()

assert len(selected) == 6
assert (selected["status"] == "COMPLETED").all()

# Verify one parameter changes between
# each consecutive configuration
hp_columns = [
    "learning_rate",
    "dropout",
    "unfrozen_layers"
]

for i in range(1, len(selected)):
    previous = selected.iloc[i - 1]
    current = selected.iloc[i]

    changes = sum(
        previous[col] != current[col]
        for col in hp_columns
    )

    assert changes == 1, (
        "Consecutive trials must differ "
        "in exactly one hyperparameter"
    )

git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

# Log each completed experiment
for _, row in selected.iterrows():

    trial_id = int(row["trial"])

    with mlflow.start_run(
        run_name=f"VisionGuard_Tuner_Trial_{trial_id}"
    ):

        mlflow.log_params({
            "architecture": "MobileNetV2",
            "learning_rate": float(row["learning_rate"]),
            "dropout": float(row["dropout"]),
            "unfrozen_layers": int(row["unfrozen_layers"]),
            "seed": 42
        })

        mlflow.log_metric(
            "best_val_loss",
            float(row["val_loss"])
        )

        mlflow.set_tags({
            "experiment_type": "hyperparameter_tuning",
            "source": "saved_keras_tuner_results",
            "logging_method": "historical_import",
            "git_commit": git_commit,
            "trial_id": str(trial_id)
        })

        mlflow.log_artifact(str(csv_path))

    print(f"Logged Trial {trial_id}")

print("\nSix tuning runs uploaded successfully!")
