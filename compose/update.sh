#!/bin/bash
set -e

IMAGE_NAME="logus2k/femulator:latest"
DOCKERFILE="femulator.Dockerfile"

echo "[$(date +%H:%M:%S)] Updating image: $IMAGE_NAME"

./stop.sh

docker ps -aq --filter "name=femulator" | xargs -r docker rm -f

docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 && docker rmi -f "$IMAGE_NAME" || true

docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" ..

echo "[$(date +%H:%M:%S)] Update complete. Run ./start.sh"
