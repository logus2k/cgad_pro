@echo off

set IMAGE_NAME=femulator:1.0
set DOCKERFILE=femulator.Dockerfile

echo Checking image: %IMAGE_NAME%

:: Check if the image exists
docker image inspect %IMAGE_NAME% >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Image '%IMAGE_NAME%' exists. Proceeding to stop containers and remove the image.

    :: Stop containers using the stop.bat script
    call stop.bat

    :: Remove the image
    docker rmi %IMAGE_NAME%
    echo Image '%IMAGE_NAME%' removed.

    :: Rebuild the image
    echo Rebuilding image '%IMAGE_NAME%'...
    docker build --no-cache -t %IMAGE_NAME% -f %DOCKERFILE% ..
    echo Image '%IMAGE_NAME%' rebuilt.
) else (
    echo Image '%IMAGE_NAME%' does not exist. Building it now...
    docker build --no-cache -t %IMAGE_NAME% -f %DOCKERFILE% ..
    echo Image '%IMAGE_NAME%' built.
)

echo Update complete. Run 'start.bat' to start the containers.
