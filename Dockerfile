# Python 3.12 rather than 3.14: torch and sentence-transformers wheels are reliably
# available here, and the image should not be the place you discover otherwise.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, from its own index. Installing it before the rest stops pip
# from pulling the ~2GB CUDA build as a transitive dependency of sentence-transformers.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the models into the image. This costs ~150MB but makes container start
# deterministic and removes a network dependency from the runtime path - a cold
# start that downloads weights is a cold start that can fail.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
RUN pip install --no-deps -e .

# The index is a mounted volume, not baked in: it changes far more often than the
# code and is far larger.
VOLUME ["/app/data"]

EXPOSE 8000

# Fails until the index is loaded and the model warmup has completed, so an
# orchestrator does not route traffic to a container that is still starting.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ragmed.api:app", "--host", "0.0.0.0", "--port", "8000"]
