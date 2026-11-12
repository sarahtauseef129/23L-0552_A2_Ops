
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "models/mobilenet_tuned_best.keras"
    )
)

THRESHOLD = float(
    os.getenv("CLASSIFICATION_THRESHOLD", "0.70")
)

if not 0 <= THRESHOLD <= 1:
    raise ValueError("Threshold must be between 0 and 1.")

print(f"Loading champion checkpoint: {MODEL_PATH}")

# Loaded only once when the application starts.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("VisionGuard model loaded successfully!")


def predict_image(image):
    """
    Accepts a PIL image and returns a prediction.
    Input preprocessing matches the training pipeline:
    RGB -> 224x224 bilinear resize -> float32 0-255.
    """

    image = image.convert("RGB")

    array = np.asarray(image, dtype=np.float32)

    # Match TensorFlow training resize behavior.
    tensor = tf.image.resize(
        array,
        [224, 224],
        method="bilinear"
    )

    batch = tf.expand_dims(tensor, axis=0)

    # Model contains its own MobileNetV2 scaling.
    probability = float(
        model.predict(batch, verbose=0).reshape(-1)[0]
    )

    if not np.isfinite(probability):
        raise ValueError("Model returned an invalid prediction.")

    if probability >= THRESHOLD:
        label = "AI-generated"
        confidence = probability
    else:
        label = "Real"
        confidence = 1 - probability

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "ai_probability": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "model_name": "MobileNetV2_Tuned"
    }
