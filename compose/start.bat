@echo off
setlocal

:: Create the network if it doesn't exist
docker network inspect femulator_network >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Creating femulator_network...
    docker network create femulator_network
)

:: Check if nvidia-smi is available on the host
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo GPU detected. Starting container with GPU support.
    docker compose -f docker-compose-gpu.yml up -d
) else (
    echo No GPU detected. Starting container without GPU support.
    docker compose -f docker-compose-cpu.yml up -d
)
