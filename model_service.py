#!/usr/bin/env python3
"""
Model service for Angular Leaf Spot Detection
Provides model loading and prediction functionality for the API server
"""

import os
import io
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AngularLeafSpotModel:
    """Service class for Angular Leaf Spot detection model"""
    
    def __init__(self, model_path="models/beenleaf_model.h5"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model"""
        if self.is_loaded:
            return True
            
        if not Path(self.model_path).exists():
            logger.error(f"Model not found at {self.model_path}")
            return False
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_data):
        """
        Preprocess image data for model prediction
        
        Args:
            image_data: Can be file path (str), numpy array, PIL Image, or bytes
            
        Returns:
            numpy array ready for model prediction or None if error
        """
        try:
            # Handle different input types
            if isinstance(image_data, str):
                # File path
                if not Path(image_data).exists():
                    logger.error(f"Image file not found: {image_data}")
                    return None
                img = cv2.imread(str(image_data))
                if img is None:
                    logger.error(f"Could not load image: {image_data}")
                    return None
                    
            elif isinstance(image_data, bytes):
                # Bytes data (from uploaded file)
                img_array = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    logger.error("Could not decode image from bytes")
                    return None
                    
            elif isinstance(image_data, Image.Image):
                # PIL Image
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
                
            elif isinstance(image_data, np.ndarray):
                # Already a numpy array
                img = image_data
                
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
            
            # Resize to model input size (256x256)
            resized = tf.image.resize(img, (256, 256))
            
            # Normalize to 0-1 range
            normalized = resized / 255.0
            
            # Add batch dimension
            input_batch = np.expand_dims(normalized, 0)
            
            return input_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_data):
        """
        Predict if leaf has angular leaf spot
        
        Args:
            image_data: Image data in various formats (path, bytes, PIL Image, numpy array)
            
        Returns:
            dict with prediction results or None if error
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        # Preprocess image
        processed_image = self.preprocess_image(image_data)
        if processed_image is None:
            return None
        
        try:
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # Interpret result
            if confidence > 0.5:
                status = "unhealthy"
                result = "Angular Leaf Spot Detected"
                health_status = "UNHEALTHY"
            else:
                status = "healthy"
                result = "Healthy Leaf"
                health_status = "HEALTHY"
            
            return {
                'status': status,
                'health_status': health_status,
                'confidence': confidence,
                'result': result,
                'threshold': 0.5,
                'interpretation': f">0.5 = unhealthy, ≤0.5 = healthy"
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def predict_batch(self, image_data_list):
        """
        Predict multiple images
        
        Args:
            image_data_list: List of image data in various formats
            
        Returns:
            list of prediction results
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        results = []
        for i, image_data in enumerate(image_data_list):
            result = self.predict(image_data)
            if result:
                result['image_index'] = i
                results.append(result)
            else:
                results.append({
                    'image_index': i,
                    'error': 'Failed to process image'
                })
        
        return results

# Global model instance for the API
model_service = AngularLeafSpotModel()