# syntax=docker/dockerfile:1

#############################
# Builder: CUDA + toolchain #
#############################
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# System deps (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.10 python3.10-venv python3-pip \
	build-essential cmake git curl ca-certificates \
	gcc-12 g++-12 \
	&& rm -rf /var/lib/apt/lists/*

# Create and activate virtualenv
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Pip up-to-date
RUN pip install --upgrade pip

# --- Python dependencies ---
RUN pip install --no-cache-dir \
	python-socketio \
	uvicorn fastapi python-socketio numpy pandas scipy matplotlib cupy_cuda13x nvidia-cublas numba-cuda h5py

# Tell CMake to build CUDA, and skip tools/tests/examples (prevents mtmd/CLI)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF" \
	FORCE_CMAKE=1

# Make CUDA driver STUB available at link time (real driver is provided at runtime)
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
	LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
	LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

# Make CUDA driver stubs visible *and* satisfy the exact SONAME libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
 
WORKDIR /fem


###########################
# Runtime: CUDA + Python  #
###########################
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal runtime deps (Python 3.10 and OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.10 python3-pip libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Bring in the venv from builder
COPY --from=builder /opt/venv /opt/venv

# Runtime env (do NOT include stubs path)
ENV PATH=/opt/venv/bin:$PATH \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

WORKDIR /fem
