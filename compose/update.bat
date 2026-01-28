@echo off
setlocal

set "IMAGE_NAME=logus2k/femulator:latest"
set "DOCKERFILE=femulator.Dockerfile"

echo [%TIME%] Updating image: %IMAGE_NAME%

:: Stop containers
call stop.bat || exit /b 1

:: Remove containers (if any)
docker ps -aq --filter "name=femulator" >nul 2>&1 && (
	docker rm -f femulator
)

:: Remove image (if exists)
docker image inspect %IMAGE_NAME% >nul 2>&1 && (
	docker rmi -f %IMAGE_NAME%
)

:: Rebuild image (cache allowed for update)
docker build -t %IMAGE_NAME% -f %DOCKERFILE% .. || exit /b 1

echo [%TIME%] Update complete. Run start.bat.
