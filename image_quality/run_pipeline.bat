@echo off
cd /d "%~dp0"
python william_image_quality_pipeline.py
if errorlevel 1 (
    echo.
    echo The pipeline encountered an error.
    pause
    exit /b 1
)
echo.
echo Completed. Open the outputs folder to view the results.
pause
