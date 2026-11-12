
import os
import io
import hmac
from pathlib import Path

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

# Keep below Vercel's request body limit
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            BASE_DIR / "models" /
            "mobilenet_tuned_best.keras"
        )
    )
)

THRESHOLD = float(
    os.getenv("CLASSIFICATION_THRESHOLD", "0.70")
)

API_KEY = os.getenv("VISIONGUARD_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "VISIONGUARD_API_KEY must be configured"
    )

if not 0 <= THRESHOLD <= 1:
    raise ValueError("Threshold must be between 0 and 1")

print("Loading VisionGuard API model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("VisionGuard API model loaded!")


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "MobileNetV2_Tuned"
    })


@app.post("/predict")
def predict():

    supplied_key = request.headers.get("X-API-Key", "")

    if not hmac.compare_digest(
        supplied_key,
        API_KEY
    ):
        return jsonify({
            "error": "Unauthorized"
        }), 401

    file = request.files.get("image")

    if file is None:
        return jsonify({
            "error": "Missing image file"
        }), 400

    try:
        with Image.open(io.BytesIO(file.read())) as img:

            if img.format not in {
                "JPEG", "PNG", "WEBP"
            }:
                return jsonify({
                    "error": "Unsupported image format"
                }), 400

            img.load()
            image = img.convert("RGB")

        array = np.asarray(image, dtype=np.float32)

        resized = tf.image.resize(
            array,
            [224, 224],
            method="bilinear"
        )

        batch = tf.expand_dims(resized, axis=0)

        ai_probability = float(
            model.predict(
                batch,
                verbose=0
            ).reshape(-1)[0]
        )

        if not np.isfinite(ai_probability):
            raise ValueError("Invalid prediction")

        if ai_probability >= THRESHOLD:
            label = "AI-generated"
            confidence = ai_probability
        else:
            label = "Real"
            confidence = 1 - ai_probability

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "ai_probability": round(ai_probability * 100, 2),
            "threshold": THRESHOLD,
            "model": "MobileNetV2_Tuned"
        })

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        return jsonify({
            "error": "Invalid or corrupt image"
        }), 400


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return jsonify({
        "error": "Image exceeds the upload limit"
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )

