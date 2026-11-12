
# 1. Lightweight Python base
FROM python:3.11-slim

# 2. Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/mobilenet_tuned_best.keras \
    CLASSIFICATION_THRESHOLD=0.70 \
    DATABASE_URL=sqlite:////app/instance/visionguard.db \
    MAX_UPLOAD_MB=10 \
    FLASK_DEBUG=false

WORKDIR /app

# 3. Install dependencies before copying app code
COPY requirements-docker.txt .

RUN pip install --no-cache-dir --no-compile \
    -r requirements-docker.txt

# 4. Create an unprivileged user
RUN groupadd --system appgroup \
    && useradd --system \
       --gid appgroup \
       --home-dir /app \
       appuser \
    && mkdir -p /app/instance /app/models \
    && chown -R appuser:appgroup /app

# 5. Copy only necessary application files
COPY --chown=appuser:appgroup app.py inference.py database.py ./
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup static/ ./static/

# 6. Run as a non-root user
USER appuser

# 7. Document the application port
EXPOSE 5001

# 8. Health check
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=5)" || exit 1

# 9. Start the Flask application through Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
