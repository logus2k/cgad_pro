# STAGE 1: Builder
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Use the native Python version for Ubuntu 24.04 (Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake git gcc-12 g++-12 \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv using the native python3
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    python-socketio uvicorn fastapi numpy pandas scipy matplotlib h5py weasyprint playwright pypandoc \
    psutil cupy-cuda13x numba-cuda cuda-python nvtx

RUN playwright install chromium

# STAGE 2: Runtime
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

# Install dependencies and Nsight tools from NVIDIA repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 \
    gnupg curl \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        nsight-systems-2025.5.2 \
        nsight-compute-2025.4.1 \
    && rm -rf /var/lib/apt/lists/*

# 1. FIX FOR CUPY: Copy CUDA headers for JIT compilation
COPY --from=builder /usr/local/cuda/include /usr/local/cuda/include

# 2. FIX FOR NUMBA: Copy NVVM for compiler support
COPY --from=builder /usr/local/cuda/nvvm /usr/local/cuda/nvvm

# Copy the virtual environment
COPY --from=builder /opt/venv /opt/venv

# Set Environment Variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda

WORKDIR /
