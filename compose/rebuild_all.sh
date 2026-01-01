#!/bin/bash

# Function to stop containers, remove image, and rebuild
rebuild_image() {
    local image_name="$1"
    local dockerfile="$2"

    echo "Checking image: $image_name"

    # Check if the image exists
    if docker image inspect "$image_name" >/dev/null 2>&1; then
        echo "Image '$image_name' exists. Proceeding to stop containers and remove the image."

        # Stop containers using the stop.sh script
        if ! ./stop.sh; then
            echo "ERROR: Failed to stop containers."
            exit 1
        fi

        # Remove the image
        if ! docker rmi "$image_name"; then
            echo "ERROR: Failed to remove image '$image_name'."
            exit 1
        fi
        echo "Image '$image_name' removed."

        # Rebuild the image
        echo "Rebuilding image '$image_name'..."
        if ! docker build --no-cache -t "$image_name" -f "$dockerfile" ..; then
            echo "ERROR: Failed to rebuild image '$image_name'."
            exit 1
        fi
        echo "Image '$image_name' rebuilt."
    else
        echo "Image '$image_name' does not exist. Building it now..."
        if ! docker build --no-cache -t "$image_name" -f "$dockerfile" ..; then
            echo "ERROR: Failed to build image '$image_name'."
            exit 1
        fi
        echo "Image '$image_name' built."
    fi
}

# Rebuild each image
rebuild_image "femulator.server:1.0" "femulator.server.Dockerfile"
rebuild_image "femulator:1.0" "femulator.Dockerfile"

echo "Rebuild complete. Run './start.sh' to start the containers."
