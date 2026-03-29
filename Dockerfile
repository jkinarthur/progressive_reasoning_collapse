# =============================================================================
# CARR: Collapse-Aware Register Recommendation
# PyTorch + CUDA base image (matches g4dn / g5 EC2 instances)
# =============================================================================
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY experiments/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full experiments codebase
COPY experiments/ /workspace/experiments/

# Create runtime directories (data downloaded at runtime, not baked in)
RUN mkdir -p /workspace/data \
             /workspace/results \
             /workspace/checkpoints \
             /workspace/figures \
             /workspace/tables

# Set PYTHONPATH so modules resolve correctly
ENV PYTHONPATH=/workspace/experiments

# Default working directory for runs
WORKDIR /workspace/experiments

# Default command: show usage
CMD ["python", "main.py", "--help"]
