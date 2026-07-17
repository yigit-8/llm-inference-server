FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Cache downloaded model weights inside the image dir the app user owns, so the
# model is fetched once and survives across restarts of the same container.
ENV HF_HOME=/app/.cache/huggingface
RUN useradd -m -u 1000 appuser && mkdir -p /app/.cache && chown -R appuser /app
USER appuser

EXPOSE 8000

# Shell form so ${PORT} expands: PaaS hosts inject PORT, local and compose do not.
CMD uvicorn src.serve:app --host 0.0.0.0 --port ${PORT:-8000}
