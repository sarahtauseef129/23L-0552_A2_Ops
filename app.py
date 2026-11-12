
import os
import io
import base64

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from inference import predict_image
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

# Initialize schema once at startup.
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
        with Image.open(file.stream) as img:
            if img.format not in {"JPEG", "PNG", "WEBP"}:
                return render_error(
                    "File content is not a supported image."
                )

            img.load()
            image = img.convert("RGB")

        prediction = predict_image(image)

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

        save_prediction(filename, prediction)

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
            "model": "loaded",
            "database": "connected"
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
        port=5001,
        debug=False
    )
