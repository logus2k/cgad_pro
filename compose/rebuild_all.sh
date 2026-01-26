#!/bin/bash
set -e

echo "[$(date +%H:%M:%S)] FEMULATOR - FULL CLEAN REBUILD"

./stop.sh

# Remove containers
docker ps -aq --filter "name=femulator" | xargs -r docker rm -f

# Remove images
docker image inspect logus2k/femulator.server:latest >/dev/null 2>&1 && docker rmi -f logus2k/femulator.server:latest || true
docker image inspect logus2k/femulator:latest        >/dev/null 2>&1 && docker rmi -f logus2k/femulator:latest        || true

docker image prune -f >/dev/null

# Rebuild images (hard, no cache)
docker build --no-cache -t logus2k/femulator.server:latest -f femulator.server.Dockerfile ..
docker build --no-cache -t logus2k/femulator:latest        -f femulator.Dockerfile        ..

echo "[$(date +%H:%M:%S)] Rebuild complete. Run ./start.sh"
