#!/bin/bash
set -e

echo "[$(date +%H:%M:%S)] FEMULATOR – FULL CLEAN REBUILD"

./stop.sh

# Remove containers
docker ps -aq --filter "name=femulator" | xargs -r docker rm -f

# Remove images
docker image inspect femulator.server:1.0 >/dev/null 2>&1 && docker rmi -f femulator.server:1.0 || true
docker image inspect femulator:1.0        >/dev/null 2>&1 && docker rmi -f femulator:1.0        || true

docker image prune -f >/dev/null

# Rebuild images (hard, no cache)
docker build --no-cache -t femulator.server:1.0 -f femulator.server.Dockerfile ..
docker build --no-cache -t femulator:1.0        -f femulator.Dockerfile        ..

echo "[$(date +%H:%M:%S)] Rebuild complete. Run ./start.sh"
