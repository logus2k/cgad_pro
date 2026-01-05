@echo off
setlocal

echo [%TIME%] Stopping FEMulator containers...

:: Stop containers using both compose files (safe & deterministic)
docker compose -f docker-compose-gpu.yml down >nul 2>&1
docker compose -f docker-compose-cpu.yml down >nul 2>&1

if %ERRORLEVEL% neq 0 (
	echo [%TIME%] ERROR: Failed to stop containers.
	exit /b 1
)

echo [%TIME%] Containers stopped successfully.
