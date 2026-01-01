@echo off
setlocal

echo ====================================
echo [%TIME%] FEMULATOR PRO - FULL REMOVE
echo ====================================

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
docker ps -aq --filter "name=femulator" >nul 2>&1 && (
	echo [%TIME%] Removing femulator containers...
	docker rm -f femulator
)

:: -------------------------------------------------
:: Remove images (if they exist)
:: -------------------------------------------------
echo [%TIME%] Removing images...

docker image inspect femulator.server:1.0 >nul 2>&1 && (
	docker rmi -f femulator.server:1.0
	echo [%TIME%] Image 'femulator.server:1.0' removed.
) || (
	echo [%TIME%] Image 'femulator.server:1.0' does not exist.
)

docker image inspect femulator:1.0 >nul 2>&1 && (
	docker rmi -f femulator:1.0
	echo [%TIME%] Image 'femulator:1.0' removed.
) || (
	echo [%TIME%] Image 'femulator:1.0' does not exist.
)

:: -------------------------------------------------
:: Optional: clean dangling layers
:: -------------------------------------------------
docker image prune -f >nul

echo =========================================
echo [%TIME%] Remove complete
echo =========================================
