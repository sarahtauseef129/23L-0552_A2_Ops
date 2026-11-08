
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Load audit results
df = pd.read_csv("reports/dataset_audit.csv")

# Remove corrupt and exact duplicate images
df = df[
    (~df["corrupt"]) &
    (~df["duplicate"])
].copy()

# Map classes to numeric labels
# 0 = Real, 1 = AI-generated
df["label"] = df["class"].map({
    "Real": 0,
    "AI-generated": 1
})

assert df["label"].notna().all(), "Unknown class found!"

# First: 80% training, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Split temporary set equally:
# 10% validation, 10% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Create output directory
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

# Save split lists
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("========== DATASET SPLITS ==========")

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images: {len(test_df)}")

print("\nClass distribution:")

for name, split in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"\n{name}:")
    print(split["class"].value_counts())

print("\nSplit files saved in data/splits/")
