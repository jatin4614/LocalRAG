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

# Wave 1b (review §10.1): drop root. model_manager is a small FastAPI
# proxy that talks to vllm + whisper over HTTP — no host filesystem
# access, no GPU, no privileged ops, so non-root is friction-free here.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} orgchat && useradd -u ${UID} -g ${GID} -m -s /bin/bash orgchat && \
    chown -R ${UID}:${GID} /app 2>/dev/null || true
USER ${UID}:${GID}

HEALTHCHECK --interval=20s --timeout=5s --retries=5 --start-period=10s \
  CMD curl -sf http://localhost:${MODEL_MANAGER_PORT}/healthz || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "model_manager.app:build_app", "--host", "0.0.0.0", "--port", "8080", "--factory"]
