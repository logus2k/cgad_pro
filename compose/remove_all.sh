#!/bin/bash

# Stop containers
./stop.sh

# Remove images
docker rmi femulator.server:1.0 femulator:1.0

echo "Containers and images removed."
