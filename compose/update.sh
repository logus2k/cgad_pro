#!/bin/bash
set -e

IMAGE_NAME="femulator:1.0"
DOCKERFILE="femulator.Dockerfile"

echo "[$(date +%H:%M:%S)] Updating image: $IMAGE_NAME"

# Stop containers
./stop.sh

# Remove containers
docker ps -aq --filter "name=femulator" | xargs -r docker rm -f

# Remove image if exists
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 && docker rmi -f "$IMAGE_NAME" || true

# Rebuild (cache allowed)
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" ..

echo "[$(date +%H:%M:%S)] Update complete. Run ./start.sh"
