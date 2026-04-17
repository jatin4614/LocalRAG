FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy as a proper package so relative imports work
RUN mkdir -p model_manager
COPY __init__.py config.py client.py tracker.py sleeper.py app.py model_manager/

ENV MODEL_MANAGER_HOST=0.0.0.0
ENV MODEL_MANAGER_PORT=8080

HEALTHCHECK --interval=20s --timeout=5s --retries=5 --start-period=10s \
  CMD curl -sf http://localhost:${MODEL_MANAGER_PORT}/healthz || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "model_manager.app:build_app", "--host", "0.0.0.0", "--port", "8080", "--factory"]
