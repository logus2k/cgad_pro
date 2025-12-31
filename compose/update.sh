#!/bin/bash

IMAGE_NAME="femulator:1.0"
DOCKERFILE="femulator.Dockerfile"

echo "Checking image: $IMAGE_NAME"

# Check if the image exists
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Image '$IMAGE_NAME' exists. Proceeding to stop containers and remove the image."

    # Stop containers using the stop.sh script
    ./stop.sh

    # Remove the image
    docker rmi "$IMAGE_NAME"
    echo "Image '$IMAGE_NAME' removed."

    # Rebuild the image
    echo "Rebuilding image '$IMAGE_NAME'..."
    docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE" ..
    echo "Image '$IMAGE_NAME' rebuilt."
else
    echo "Image '$IMAGE_NAME' does not exist. Building it now..."
    docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE" ..
    echo "Image '$IMAGE_NAME' built."
fi

echo "Update complete. Run './start.sh' to start the containers."
