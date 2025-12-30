FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    build-essential cmake git curl ca-certificates \
    gcc-12 g++-12 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtualenv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pip up-to-date
RUN pip install --upgrade pip

# Python dependencies
RUN pip install --no-cache-dir \
	nvidia-cuda-nvcc nvidia-cuda-runtime \
    python-socketio uvicorn fastapi numpy pandas scipy matplotlib h5py \
    cupy-cuda13x numba-cuda cuda-python

# CUDA environment
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda

WORKDIR /
