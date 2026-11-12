
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)

MODEL_URI = "models:/VisionGuard_Classifier@champion"

if len(sys.argv) < 2:
    print("Usage: python scripts/predict_champion.py IMAGE_PATH")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise FileNotFoundError(image_path)

print("Loading champion model from DagsHub...")

model = mlflow.tensorflow.load_model(MODEL_URI)

# Same input format as training:
# RGB, 224x224, pixel values from 0 to 255.
# The model already includes MobileNetV2 rescaling.
with Image.open(image_path) as img:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    image = np.asarray(img, dtype=np.float32)

image = np.expand_dims(image, axis=0)

prob_ai = float(model.predict(image, verbose=0)[0][0])

threshold = 0.5

if prob_ai >= threshold:
    label = "AI-generated"
    confidence = prob_ai
else:
    label = "Real"
    confidence = 1 - prob_ai

print("\n========== PREDICTION ==========")
print("Image:", image_path.name)
print("Model:", MODEL_URI)
print("Prediction:", label)
print(f"Confidence: {confidence * 100:.2f}%")
print(f"AI probability: {prob_ai:.4f}")
