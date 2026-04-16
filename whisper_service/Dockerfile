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

HEALTHCHECK --interval=20s --timeout=5s --retries=10 --start-period=60s \
  CMD curl -sf http://localhost:${PORT}/health || exit 1

EXPOSE 9000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
