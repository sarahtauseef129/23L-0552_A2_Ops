
from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

records = []

classes = {
    "AiArtData": "AI-generated",
    "RealArt": "Real"
}

def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


for folder, label in classes.items():
    class_path = DATA_DIR / folder

    if not class_path.exists():
        raise FileNotFoundError(f"Missing folder: {class_path}")

    for image_path in class_path.rglob("*"):
        if not image_path.is_file():
            continue

        record = {
            "filepath": str(image_path),
            "class": label,
            "file_size_kb": round(
                image_path.stat().st_size / 1024, 2
            ),
            "width": None,
            "height": None,
            "color_mode": None,
            "format": None,
            "sha256": calculate_hash(image_path),
            "corrupt": False
        }

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image.load()
                record["width"] = image.width
                record["height"] = image.height
                record["color_mode"] = image.mode
                record["format"] = image.format

        except Exception:
            record["corrupt"] = True

        records.append(record)


df = pd.DataFrame(records)

df["duplicate"] = df.duplicated(
    subset="sha256", keep="first"
)

df.to_csv(
    OUTPUT_DIR / "dataset_audit.csv",
    index=False
)

valid_df = df[~df["corrupt"]]

print("\n========== DATASET AUDIT ==========")

print("\nImages per class:")
print(df.groupby("class").size())

print("\nStatistics per class:")
print(
    valid_df.groupby("class")[
        ["width", "height", "file_size_kb"]
    ].agg(["min", "median", "max"]).round(2)
)

print("\nCorrupt files:", df["corrupt"].sum())
print("Exact duplicate copies:", df["duplicate"].sum())

print("\nImage formats:")
print(valid_df.groupby(["class", "format"]).size())

print("\nColor modes:")
print(valid_df.groupby(["class", "color_mode"]).size())

print("\nAudit saved to reports/dataset_audit.csv")
