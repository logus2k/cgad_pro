@echo off

:: List of images to check and rebuild
set IMAGES[0]=femulator.server:1.0 femulator.server.Dockerfile
set IMAGES[1]=femulator:1.0 femulator.Dockerfile

:: Function to stop containers, remove image, and rebuild
for %%i in (0 1) do (
    for /f "tokens=1,2" %%a in ("!IMAGES[%%i]!") do (
        set "image_name=%%a"
        set "dockerfile=%%b"

        echo Checking image: !image_name!

        :: Check if the image exists
        docker image inspect !image_name! >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo Image '!image_name!' exists. Proceeding to stop containers and remove the image.

            :: Stop containers using the stop.bat script
            call stop.bat

            :: Remove the image
            docker rmi !image_name!
            echo Image '!image_name!' removed.

            :: Rebuild the image
            echo Rebuilding image '!image_name!'...
            docker build --no-cache -t !image_name! -f !dockerfile! ..
            echo Image '!image_name!' rebuilt.
        ) else (
            echo Image '!image_name!' does not exist. Building it now...
            docker build --no-cache -t !image_name! -f !dockerfile! ..
            echo Image '!image_name!' built.
        )
    )
)

echo Rebuild complete. Run 'start.bat' to start the containers.
