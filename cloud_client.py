
import io
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def predict_using_cloud(image_bytes, filename):
    api_url = os.getenv("VISIONGUARD_API_URL")
    api_key = os.getenv("VISIONGUARD_API_KEY")

    if not api_url or not api_key:
        raise RuntimeError(
            "Cloud API URL or key is missing from .env"
        )

    response = requests.post(
        api_url,
        headers={
            "X-API-Key": api_key
        },
        files={
            "image": (
                filename,
                io.BytesIO(image_bytes),
                "application/octet-stream"
            )
        },
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    return {
        "label": result["prediction"],
        "confidence": result["confidence"],
        "ai_probability": result["ai_probability"],
        "model_name": result["model"]
    }
