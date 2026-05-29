# ============================================================================
# 🐳 FinanceRAG Docker Image
# ============================================================================
# Multi-stage build for optimized image size
# Based on actual working environment (Python 3.10)

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 1: Install Python dependencies
# ============================================================================
FROM base as builder

# Copy only requirements first (better layer caching)
COPY requirements.txt ./

# Install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================================================
# Stage 2: Final image
# ============================================================================
FROM base

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/output \
    /root/.cache/huggingface

# Set Python path
ENV PYTHONPATH=/workspace

# Expose Jupyter port
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
