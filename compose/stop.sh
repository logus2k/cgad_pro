#!/bin/bash

# Stop the Docker Compose service
echo "Stopping the FEMulator Pro application..."
docker compose down

# Check the exit status of the command
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to stop the FEMulator Pro application."
    exit 1
fi

# Check if the container is still running
if docker compose ps | grep -q "femulator"; then
    echo "WARNING: The FEMulator Pro container is still running. Attempt to stop it manually or check logs."
    exit 1
fi

# Verify if the container exists at all (running or stopped)
if ! docker compose ps -a | grep -q "femulator"; then
    echo "The FEMulator Pro application is not running."
else
    echo "The FEMulator Pro container was not running, but it has been removed."
fi

echo "Execute ./start.sh when you wish to launch the application."
