@echo off
REM Angular Leaf Spot Detection API Server Startup Script (Windows)
REM Usage: start_server.bat [port] [debug]

REM Default values
set DEFAULT_PORT=5000
set DEFAULT_DEBUG=false

REM Get parameters
if "%1"=="" (
    set PORT=%DEFAULT_PORT%
) else (
    set PORT=%1
)

if "%2"=="" (
    set DEBUG=%DEFAULT_DEBUG%
) else (
    set DEBUG=%2
)

echo 🌿 Angular Leaf Spot Detection API Server
echo ==========================================

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Consider creating one:
    echo    python -m venv venv
    echo    venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
)

REM Check if requirements are installed
echo 🔍 Checking dependencies...
python -c "import flask, tensorflow, cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo ❌ Missing dependencies. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies. Please check requirements.txt
        exit /b 1
    )
) else (
    echo ✅ Dependencies are satisfied
)

REM Check if model exists
if not exist "models\beenleaf_model.h5" (
    echo ⚠️  Model file not found at models\beenleaf_model.h5
    echo    Please ensure the model is trained and saved in the correct location.
    echo    The API will start but predictions will fail until the model is available.
)

REM Set environment variables
set PORT=%PORT%
set DEBUG=%DEBUG%

echo 🚀 Starting API server...
echo    Port: %PORT%
echo    Debug: %DEBUG%
echo    URL: http://localhost:%PORT%
echo.
echo 📋 Available endpoints:
echo    GET  http://localhost:%PORT%/                - Health check
echo    GET  http://localhost:%PORT%/model/info      - Model information
echo    POST http://localhost:%PORT%/predict         - Predict single image
echo    POST http://localhost:%PORT%/predict/batch   - Predict multiple images
echo    POST http://localhost:%PORT%/predict/url     - Predict image from URL
echo.
echo Press Ctrl+C to stop the server
echo ==========================================

REM Start the server
python api_server.py