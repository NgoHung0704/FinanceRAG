# ============================================================================
# 💹 FinanceRAG — Streamlit app image (Retrieve -> Rerank -> Generate)
# ============================================================================
# Build:  docker build -f Dockerfile.app -t financerag-app .
# Run:    docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... \
#                 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" financerag-app

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first for better layer caching.
COPY requirements-app.txt ./
RUN pip install --upgrade pip && pip install -r requirements-app.txt

# App code (data/ and models/ are mounted at runtime, not baked in).
COPY financerag_app/ ./financerag_app/
COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
