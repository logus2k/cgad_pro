#!/bin/bash
set -e

echo "[$(date +%H:%M:%S)] FEMULATOR - FULL REMOVE"

./stop.sh

# Remove containers (if any)
docker ps -aq --filter "name=femulator" | xargs -r docker rm -f

# Remove images (if present)
docker image inspect femulator.server:1.0 >/dev/null 2>&1 && docker rmi -f femulator.server:1.0 || true
docker image inspect femulator:1.0        >/dev/null 2>&1 && docker rmi -f femulator:1.0        || true

docker image prune -f >/dev/null

echo "[$(date +%H:%M:%S)] Remove complete."
