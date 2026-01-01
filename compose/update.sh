#!/bin/bash

IMAGE_NAME="femulator:1.0"
DOCKERFILE="femulator.Dockerfile"

echo "Checking image: $IMAGE_NAME"

# Check if the image exists
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Image '$IMAGE_NAME' exists. Proceeding to stop containers and remove the image."

    # Stop containers using the stop.sh script
    if ! ./stop.sh; then
        echo "ERROR: Failed to stop containers."
        exit 1
    fi

    # Remove the image
    if ! docker rmi "$IMAGE_NAME"; then
        echo "ERROR: Failed to remove image '$IMAGE_NAME'."
        exit 1
    fi
    echo "Image '$IMAGE_NAME' removed."

    # Rebuild the image
    echo "Rebuilding image '$IMAGE_NAME'..."
    if ! docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE" ..; then
        echo "ERROR: Failed to rebuild image '$IMAGE_NAME'."
        exit 1
    fi
    echo "Image '$IMAGE_NAME' rebuilt."
else
    echo "Image '$IMAGE_NAME' does not exist. Building it now..."
    if ! docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE" ..; then
        echo "ERROR: Failed to build image '$IMAGE_NAME'."
        exit 1
    fi
    echo "Image '$IMAGE_NAME' built."
fi

echo "Update complete. Run './start.sh' to start the containers."
