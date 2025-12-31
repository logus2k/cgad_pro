@echo off

:: Stop the Docker Compose service
docker compose down femulator

:: Check the exit status of the command
if %ERRORLEVEL% equ 0 (
    :: Check if the container is still running
    docker ps | findstr "femulator" >nul
    if %ERRORLEVEL% equ 0 (
        echo WARNING: The container is still running. Attempt to stop it again or check logs.
        exit /b 1
    ) else (
        :: Verify if the container exists at all (running or stopped)
        docker ps -a | findstr "femulator" >nul
        if %ERRORLEVEL% neq 0 (
            echo The FEMulator Pro application is not running.
            echo Execute start.bat when you wish to launch the application.
        ) else (
            echo The container was not running, but it has been removed.
            echo Execute start.bat when you wish to launch the application.
        )
    )
) else (
    echo ERROR: The container remove command failed.
    exit /b 1
)
