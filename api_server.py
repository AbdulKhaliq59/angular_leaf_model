#!/usr/bin/env python3
"""
Flask API Server for Angular Leaf Spot Detection
Provides REST API endpoints for image classification
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import logging
from pathlib import Path

from model_service import model_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename:
        return False
    return Path(filename).suffix.lower() in app.config['UPLOAD_EXTENSIONS']

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        allowed_exts = ', '.join(app.config['UPLOAD_EXTENSIONS'])
        return False, f"File type not allowed. Supported formats: {allowed_exts}"
    
    return True, None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Angular Leaf Spot Detection API',
        'version': '1.0.0',
        'model_loaded': model_service.is_loaded
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        # Try to load model if not already loaded
        if not model_service.is_loaded:
            model_service.load_model()
        
        return jsonify({
            'model_path': model_service.model_path,
            'model_loaded': model_service.is_loaded,
            'input_size': [256, 256, 3],
            'classes': ['healthy', 'unhealthy'],
            'threshold': 0.5,
            'description': 'Angular Leaf Spot Detection Model'
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Predict single image endpoint
    Expects form data with 'image' file field
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Validate file
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Read file data
        file_data = file.read()
        
        # Make prediction
        result = model_service.predict(file_data)
        
        if result is None:
            return jsonify({'error': 'Failed to process image or make prediction'}), 500
        
        # Add metadata
        result['filename'] = secure_filename(file.filename)
        result['file_size'] = len(file_data)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple images endpoint
    Expects form data with multiple 'images' file fields
    """
    try:
        # Check if image files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No image files provided'}), 400
        
        results = []
        
        for i, file in enumerate(files):
            # Validate each file
            is_valid, error_msg = validate_image_file(file)
            if not is_valid:
                results.append({
                    'index': i,
                    'filename': file.filename,
                    'error': error_msg
                })
                continue
            
            try:
                # Read file data
                file_data = file.read()
                
                # Make prediction
                result = model_service.predict(file_data)
                
                if result is None:
                    results.append({
                        'index': i,
                        'filename': secure_filename(file.filename),
                        'error': 'Failed to process image'
                    })
                else:
                    result['index'] = i
                    result['filename'] = secure_filename(file.filename)
                    result['file_size'] = len(file_data)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing file {i}: {e}")
                results.append({
                    'index': i,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_files': len(files),
            'results': results
        })
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'Files too large. Maximum total size is 16MB'}), 413
    except Exception as e:
        logger.error(f"Error in batch predict endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/predict/url', methods=['POST'])
def predict_from_url():
    """
    Predict image from URL
    Expects JSON with 'url' field
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url']
        
        # Import requests here to avoid dependency if not using this endpoint
        import requests
        
        # Download image from URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return jsonify({'error': 'URL does not point to an image'}), 400
        
        # Make prediction
        result = model_service.predict(response.content)
        
        if result is None:
            return jsonify({'error': 'Failed to process image or make prediction'}), 500
        
        # Add metadata
        result['source_url'] = url
        result['file_size'] = len(response.content)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download image: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in URL predict endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Load model at module level so it is available when served by gunicorn
logger.info("Loading Angular Leaf Spot model...")
if model_service.load_model():
    logger.info("Model loaded successfully!")
else:
    logger.warning("Model failed to load. API will still start but predictions will fail.")

if __name__ == '__main__':
    print("🌿 Angular Leaf Spot Detection API Server")
    print("=" * 50)
    
    # Get port from environment or default to 5001
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting server on port {port}")
    print(f"📋 Available endpoints:")
    print(f"   GET  /                - Health check")
    print(f"   GET  /model/info      - Model information")
    print(f"   POST /predict         - Predict single image (form data)")
    print(f"   POST /predict/batch   - Predict multiple images (form data)")
    print(f"   POST /predict/url     - Predict image from URL (JSON)")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=debug)