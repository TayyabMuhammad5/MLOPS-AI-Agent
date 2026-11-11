FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION="v2.1.0" \
    MODEL_NAME="churn-classifier" \
    MODEL_VERSION="2" \
    DVC_DATASET_VERSION="v2.1"

COPY pyproject.toml README.md ./
COPY agent/ ./agent/
COPY api/ ./api/
COPY dags/ ./dags/
COPY training/ ./training/
COPY data/ ./data/
COPY dvc.yaml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn pandas numpy

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
