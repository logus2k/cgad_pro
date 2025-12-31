#!/bin/bash

# Run the Docker Compose command
docker compose up -d femulator

# Check the exit status of the command
if [ $? -eq 0 ]; then
    echo "The FEMulator Pro application was successfully STARTED."

    # Verify if the container is running
    if docker ps | grep -q "femulator"; then
        echo "You can now access it at http://localhost:5867 using a browser."
        echo "Run ./stop.sh when you wish to stop the application."
    else
        echo "WARNING: The container is not running."
        exit 1
    fi
else
    echo "ERROR: The container launch command failed."
    exit 1
fi
