@echo off
setlocal

echo ================================================
echo [%TIME%] FEMULATOR PRO - FULL CLEAN REBUILD
echo ================================================

:: -------------------------------------------------
:: Stop containers
:: -------------------------------------------------
call stop.bat || (
	echo [%TIME%] ERROR: Failed to stop containers.
	exit /b 1
)

:: -------------------------------------------------
:: Remove femulator containers (if any)
:: -------------------------------------------------
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=femulator"') do (
	echo [%TIME%] Removing femulator containers...
	docker rm -f %%i
)

:: -------------------------------------------------
:: Image 1: femulator.server:latest
:: -------------------------------------------------
set "image_name=logus2k/femulator.server:latest"
set "dockerfile=femulator.server.Dockerfile"

echo [%TIME%] Checking image: %image_name%
docker image inspect %image_name% >nul 2>&1
if %ERRORLEVEL% equ 0 (
	echo [%TIME%] Removing image '%image_name%'...
	docker rmi -f %image_name%
)

echo [%TIME%] Building image '%image_name%'...
docker build --no-cache -t %image_name% -f %dockerfile% .. || exit /b 1

:: -------------------------------------------------
:: Image 2: femulator:latest
:: -------------------------------------------------
set "image_name=logus2k/femulator:latest"
set "dockerfile=femulator.Dockerfile"

echo [%TIME%] Checking image: %image_name%
docker image inspect %image_name% >nul 2>&1
if %ERRORLEVEL% equ 0 (
	echo [%TIME%] Removing image '%image_name%'...
	docker rmi -f %image_name%
)

echo [%TIME%] Building image '%image_name%'...
docker build --no-cache -t %image_name% -f %dockerfile% .. || exit /b 1

echo =========================================
echo [%TIME%] Rebuild complete. Run 'start.bat'
echo =========================================
