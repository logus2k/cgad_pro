@echo off

:: Run the Docker Compose command
docker compose up -d femulator

:: Check the exit status of the command
if %ERRORLEVEL% equ 0 (
    echo The FEMulator Pro application was successfully STARTED.

    :: Verify if the container is running
    docker ps | findstr "femulator" >nul
    if %ERRORLEVEL% equ 0 (
        echo You can now access it at http://localhost:5867 using a browser.
        echo Run stop.bat when you wish to stop the application.
    ) else (
        echo WARNING: The container is not running.
        exit /b 1
    )
) else (
    echo ERROR: The container launch command failed.
    exit /b 1
)
