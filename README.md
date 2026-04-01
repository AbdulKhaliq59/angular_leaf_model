---
<<<<<<< HEAD
title: Angular Leaf Spot Detection
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Angular Leaf Spot Detection API

A Flask-based REST API for detecting Angular Leaf Spot disease in plant leaves using deep learning.

## Overview

This API provides endpoints for image classification that can detect whether a leaf image shows signs of Angular Leaf Spot disease. The model returns predictions with confidence scores and can handle single images, batch processing, and URL-based image analysis.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Server

**Linux/Mac:**
```bash
./start_server.sh [port] [debug]
```

**Windows:**
```batch
start_server.bat [port] [debug]
```

**Direct Python:**
```bash
python api_server.py
```

**Environment Variables:**
```bash
export PORT=5001        # Server port (default: 5001)
export DEBUG=true       # Debug mode (default: false)
python api_server.py
```

### 3. Test the API

The server will start at `http://localhost:5001` (or your specified port).

## API Endpoints

### Health Check
```http
GET /
```

Returns server status and model information.

**Response:**
```json
{
  "status": "healthy",
  "service": "Angular Leaf Spot Detection API",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Model Information
```http
GET /model/info
```

Returns detailed model information.

**Response:**
```json
{
  "model_path": "models/beenleaf_model.h5",
  "model_loaded": true,
  "input_size": [256, 256, 3],
  "classes": ["healthy", "unhealthy"],
  "threshold": 0.5,
  "description": "Angular Leaf Spot Detection Model"
}
```

### Single Image Prediction
```http
POST /predict
Content-Type: multipart/form-data
```

Upload a single image for prediction.

**Request:**
- Form data with `image` field containing the image file
- Supported formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP
- Maximum file size: 16MB

**Example using curl:**
```bash
curl -X POST \
  http://localhost:5001/predict \
  -F "image=@path/to/your/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "status": "healthy",
    "health_status": "HEALTHY",
    "confidence": 0.2341,
    "result": "Healthy Leaf",
    "threshold": 0.5,
    "interpretation": ">0.5 = unhealthy, ≤0.5 = healthy",
    "filename": "image.jpg",
    "file_size": 245760
  }
}
```

### Batch Image Prediction
```http
POST /predict/batch
Content-Type: multipart/form-data
```

Upload multiple images for batch prediction.

**Request:**
- Form data with multiple `images` fields containing image files

**Example using curl:**
```bash
curl -X POST \
  http://localhost:5001/predict/batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

**Response:**
```json
{
  "success": true,
  "total_files": 2,
  "results": [
    {
      "status": "healthy",
      "health_status": "HEALTHY",
      "confidence": 0.2341,
      "result": "Healthy Leaf",
      "threshold": 0.5,
      "interpretation": ">0.5 = unhealthy, ≤0.5 = healthy",
      "index": 0,
      "filename": "image1.jpg",
      "file_size": 245760
    },
    {
      "status": "unhealthy",
      "health_status": "UNHEALTHY",
      "confidence": 0.8234,
      "result": "Angular Leaf Spot Detected",
      "threshold": 0.5,
      "interpretation": ">0.5 = unhealthy, ≤0.5 = healthy",
      "index": 1,
      "filename": "image2.jpg",
      "file_size": 189234
    }
  ]
}
```

### URL-based Prediction
```http
POST /predict/url
Content-Type: application/json
```

Predict an image from a URL.

**Request:**
```json
{
  "url": "https://example.com/leaf-image.jpg"
}
```

**Example using curl:**
```bash
curl -X POST \
  http://localhost:5001/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/leaf-image.jpg"}'
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "status": "healthy",
    "health_status": "HEALTHY",
    "confidence": 0.2341,
    "result": "Healthy Leaf",
    "threshold": 0.5,
    "interpretation": ">0.5 = unhealthy, ≤0.5 = healthy",
    "source_url": "https://example.com/leaf-image.jpg",
    "file_size": 245760
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input (no file, wrong format, etc.)
- `413 Payload Too Large`: File exceeds 16MB limit
- `404 Not Found`: Endpoint not found
- `405 Method Not Allowed`: Wrong HTTP method
- `500 Internal Server Error`: Server or model errors

**Error Response Format:**
```json
{
  "error": "Error description"
}
```

## Model Information

- **Input Size**: 256x256x3 (RGB images)
- **Classes**: `healthy`, `unhealthy`
- **Threshold**: 0.5 (confidence > 0.5 = unhealthy)
- **Model File**: `models/beenleaf_model.h5`

## File Structure

```
├── api_server.py           # Main Flask API server
├── model_service.py        # Model loading and prediction logic
├── requirements.txt        # Python dependencies
├── start_server.sh         # Linux/Mac startup script
├── start_server.bat        # Windows startup script
├── README.md              # This documentation
├── models/
│   └── beenleaf_model.h5  # Trained model file
└── data/                  # Training data (optional)
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 5001)
- `DEBUG`: Debug mode (default: false)

### Application Configuration

In `api_server.py`:
- `MAX_CONTENT_LENGTH`: Maximum file size (16MB)
- `UPLOAD_EXTENSIONS`: Allowed file extensions

## Deployment

### Development
```bash
python api_server.py
```

### Production (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 api_server:app
```

### Docker (optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001

CMD ["python", "api_server.py"]
```

## Troubleshooting

1. **Model not found**: Ensure `models/beenleaf_model.h5` exists
2. **Import errors**: Install all requirements with `pip install -r requirements.txt`
3. **Permission denied**: Make startup script executable with `chmod +x start_server.sh`
4. **Port already in use**: Change port with `PORT=8080 python api_server.py`
5. **Out of memory**: Reduce batch size or image resolution

## Performance Tips

1. **GPU Support**: Install `tensorflow-gpu` for faster predictions
2. **Model Caching**: Model is loaded once and cached in memory
3. **Batch Processing**: Use `/predict/batch` for multiple images
4. **Image Optimization**: Resize images to 256x256 before upload for faster processing
=======
title: Angular Leaf Model
emoji: 🔥
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 7b7fc67 (initial commit)
