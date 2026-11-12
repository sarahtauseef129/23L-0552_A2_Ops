
import os
import io
import base64
import requests

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from cloud_client import predict_using_cloud

from database import (
    init_db,
    save_prediction,
    get_history,
    check_database
)

load_dotenv()

app = Flask(__name__)

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Initialize database schema when Flask starts
init_db()


def render_error(message, status=400):
    return render_template(
        "index.html",
        result=None,
        error=message
    ), status


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        error=None
    )


@app.route("/result", methods=["POST"])
def result():
    file = request.files.get("image")

    if file is None or not file.filename:
        return render_error("Please select an image.")

    filename = secure_filename(file.filename)

    if not filename:
        return render_error("Invalid filename.")

    if not any(
        filename.lower().endswith(ext)
        for ext in ALLOWED_EXTENSIONS
    ):
        return render_error(
            "Only JPG, JPEG, PNG and WEBP images are allowed."
        )

    try:
        # Read the uploaded image
        image_bytes = file.read()

        # Validate the image before sending it to Vercel
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format not in {"JPEG", "PNG", "WEBP"}:
                return render_error(
                    "File content is not a supported image."
                )

            img.load()
            image = img.convert("RGB")

        # Send image to the cloud-hosted model on Vercel
        prediction = predict_using_cloud(
            image_bytes,
            filename
        )

        # Generate image preview for the website
        preview = image.copy()
        preview.thumbnail((600, 450))

        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")

        prediction["image_url"] = (
            "data:image/png;base64,"
            + base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
        )

        # Save the prediction to the existing database
        save_prediction(filename, prediction)

        # Display prediction on the website
        return render_template(
            "index.html",
            result=prediction,
            error=None
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        return render_error(
            "This image could not be decoded or processed."
        )

    except requests.exceptions.RequestException:
        app.logger.exception("Cloud API request failed")
        return render_error(
            "Cloud prediction service is unavailable. "
            "Please try again.",
            502
        )

    except (KeyError, TypeError):
        app.logger.exception("Invalid cloud API response")
        return render_error(
            "The cloud API returned an unexpected response.",
            502
        )


@app.route("/history", methods=["GET"])
def history():
    return render_template(
        "history.html",
        records=get_history()
    )


@app.route("/health", methods=["GET"])
def health():
    try:
        check_database()

        return jsonify({
            "status": "healthy",
            "inference": "cloud",
            "database": "connected",
            "model": "MobileNetV2_Tuned"
        }), 200

    except Exception:
        return jsonify({
            "status": "unhealthy",
            "database": "unavailable"
        }), 503


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return render_error(
        f"Image exceeds the {MAX_MB} MB upload limit.",
        413
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5002,
        debug=False
    )
