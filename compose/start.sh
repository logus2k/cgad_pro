#!/bin/bash
set -e

# -------------------------------------------------
# Ensure network exists
# -------------------------------------------------
if ! docker network inspect femulator_network >/dev/null 2>&1; then
	echo "Creating femulator_network..."
	docker network create femulator_network
fi

# -------------------------------------------------
# Detect usable NVIDIA GPU via Docker (authoritative)
# -------------------------------------------------
if docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
	echo "Usable NVIDIA GPU detected. Starting with GPU support."
	docker compose -f docker-compose-gpu.yml up -d
else
	echo "No usable NVIDIA GPU detected. Starting without GPU support."
	docker compose -f docker-compose-cpu.yml up -d
fi

echo "Containers started successfully."
