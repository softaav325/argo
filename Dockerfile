# Stage 1: Build & Train
FROM python:3.9-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-build.txt .

# Увеличенный таймаут для стабильности
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements-build.txt

COPY app/train.py .
COPY app/model_wrapper.py .

RUN python train.py

# Stage 2: Runtime
FROM python:3.9-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-run.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements-run.txt

COPY app/gradio_app.py .
COPY app/model_wrapper.py .

COPY --from=builder /build/model_artifacts ./model_artifacts

RUN useradd -m -u 1001 appuser \
    && chown -R 1001:1001 /app

USER appuser

EXPOSE 7860

CMD ["python", "gradio_app.py"]