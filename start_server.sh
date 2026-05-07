#!/bin/bash

# Angular Leaf Spot Detection API Server Startup Script
# Usage: ./start_server.sh [port] [debug]

# Default values
DEFAULT_PORT=5001
DEFAULT_DEBUG=false

# Get parameters
PORT=${1:-$DEFAULT_PORT}
DEBUG=${2:-$DEFAULT_DEBUG}

echo "🌿 Angular Leaf Spot Detection API Server"
echo "=========================================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
else
    echo "⚠️  No virtual environment found. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python -c "import flask, tensorflow, cv2, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check requirements.txt"
        exit 1
    fi
else
    echo "✅ Dependencies are satisfied"
fi

# Check if model exists
if [ ! -f "models/beenleaf_model.h5" ]; then
    echo "⚠️  Model file not found at models/beenleaf_model.h5"
    echo "   Please ensure the model is trained and saved in the correct location."
    echo "   The API will start but predictions will fail until the model is available."
fi

# Set environment variables
export PORT=$PORT
export DEBUG=$DEBUG

echo "🚀 Starting API server..."
echo "   Port: $PORT"
echo "   Debug: $DEBUG"
echo "   URL: http://localhost:$PORT"
echo ""
echo "📋 Available endpoints:"
echo "   GET  http://localhost:$PORT/                - Health check"
echo "   GET  http://localhost:$PORT/model/info      - Model information"
echo "   POST http://localhost:$PORT/predict         - Predict single image"
echo "   POST http://localhost:$PORT/predict/batch   - Predict multiple images"
echo "   POST http://localhost:$PORT/predict/url     - Predict image from URL"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="

# Start the server
python api_server.py