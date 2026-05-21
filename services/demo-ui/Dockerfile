FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so first request is fast and offline-safe.
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBED_MODEL=$EMBED_MODEL
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

COPY app.py .
COPY templates ./templates

ENV DEMO_UI_HOST=0.0.0.0 \
    DEMO_UI_PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app:app --host ${DEMO_UI_HOST} --port ${DEMO_UI_PORT}"]
