FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV WHISPER_MODEL=medium
ENV WHISPER_CACHE=/models
ENV PORT=9000

# Wave 1b (review §10.1): drop root. whisper bind-mounts volumes/models
# at /models for the WHISPER_CACHE path; that host dir is operator-
# managed (UID 1000 by convention). Pre-create /models so chown lands
# even if compose hasn't bound it yet at build time.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} orgchat && useradd -u ${UID} -g ${GID} -m -s /bin/bash orgchat && \
    mkdir -p /models && \
    chown -R ${UID}:${GID} /app /models 2>/dev/null || true
USER ${UID}:${GID}

HEALTHCHECK --interval=20s --timeout=5s --retries=10 --start-period=60s \
  CMD curl -sf http://localhost:${PORT}/health || exit 1

EXPOSE 9000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
