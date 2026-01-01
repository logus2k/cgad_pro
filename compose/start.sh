#!/bin/bash

# Create the network if it doesn't exist
if ! docker network inspect femulator_network >/dev/null 2>&1; then
    echo "Creating femulator_network..."
    if ! docker network create femulator_network; then
        echo "ERROR: Failed to create femulator_network."
        exit 1
    fi
fi

# Check if nvidia-smi is available on the host
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Starting container with GPU support."
    if ! docker compose -f docker-compose.gpu.yml up -d; then
        echo "ERROR: Failed to start container with GPU support."
        exit 1
    fi
else
    echo "No GPU detected. Starting container without GPU support."
    if ! docker compose -f docker-compose.cpu.yml up -d; then
        echo "ERROR: Failed to start container without GPU support."
        exit 1
    fi
fi

echo "Containers started successfully."
