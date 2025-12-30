#!/bin/bash

# Stop the Docker Compose service
docker compose down femulator

# Check the exit status of the command
if [ $? -eq 0 ]; then
    # Check if the container was running before the command
    if docker ps | grep -q "femulator"; then
        echo "WARNING: The container is still running. Attempt to stop it again or check logs."
        exit 1
    else
        # Verify if the container exists at all (running or stopped)
        if ! docker ps -a | grep -q "femulator"; then
            echo "The FEMulator Pro application is not running."
            echo "Execute ./start.sh when you wish to launch the application."
        else
            echo "The container was not running, but it has been removed."
            echo "Execute ./start.sh when you wish to launch the application."
        fi
    fi
else
    echo "ERROR: The container remove command failed."
    exit 1
fi
