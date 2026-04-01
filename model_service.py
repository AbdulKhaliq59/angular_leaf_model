#!/usr/bin/env python3
"""
<<<<<<< HEAD
Model service for Angular Leaf Spot Detection.

Multi-stage validation pipeline:
  Stage 1 — Image quality   : blur score (Laplacian) + minimum size
  Stage 2 — Leaf presence   : HSV color analysis (green / yellow-green range)
  Stage 3 — Bean-leaf shape : contour solidity + elongation ratio
  Stage 4 — Disease class.  : ML model inference (sigmoid threshold)

predicted_class values:
  'low_quality'       : blurry or too small to analyse
  'not_leaf'          : no plant-coloured pixels detected
  'not_bean_leaf'     : leaf detected but shape/colour not consistent with bean leaves
  'healthy'           : bean leaf, sigmoid <= HEALTHY_THRESHOLD
  'angular_leaf_spot' : bean leaf, sigmoid >= ALS_THRESHOLD
  'other_disease'     : bean leaf, sigmoid in uncertainty band

Class mapping (empirically confirmed):
  Class 0 = healthy           → sigmoid output ≈ 0  (low sigmoid)
  Class 1 = angular_leaf_spot → sigmoid output ≈ 1  (high sigmoid)
=======
Model service for Angular Leaf Spot Detection
Provides model loading and prediction functionality for the API server
>>>>>>> c8c3b73 (feat: Implementing Angular leaf model)
"""

import os
import io
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import logging

<<<<<<< HEAD
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Disease classification thresholds ────────────────────────────────────────
ALS_THRESHOLD     = 0.96   # sigmoid >= this → angular_leaf_spot
HEALTHY_THRESHOLD = 0.88   # sigmoid <= this → healthy
# 0.88 < sigmoid < 0.96    → other_disease (uncertain band)

# ── Stage 1 — image quality ───────────────────────────────────────────────────
BLUR_THRESHOLD   = 60.0    # Laplacian variance; below this → low_quality
MIN_IMAGE_PIXELS = 100     # minimum dimension (width or height) in pixels

# ── Stage 2 — leaf colour detection ──────────────────────────────────────────
LEAF_COLOR_RATIO_MIN = 0.08   # fraction of plant-coloured pixels required

# ── Stage 3 — bean-leaf shape validation ─────────────────────────────────────
BEAN_LEAF_MIN_SOLIDITY   = 0.35   # contour area / convex-hull area
BEAN_LEAF_MAX_ELONGATION = 7.0    # long-side / short-side bounding-box ratio


# ── Stage 1 helpers ───────────────────────────────────────────────────────────

def _check_image_quality(img: np.ndarray) -> tuple:
    """
    Returns (is_good_quality: bool, reason: str).
    Checks minimum dimensions and Laplacian blur score.
    """
    try:
        h, w = img.shape[:2]
        if min(h, w) < MIN_IMAGE_PIXELS:
            return False, f'image too small ({w}x{h} px)'

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < BLUR_THRESHOLD:
            return False, f'image too blurry (score={blur_score:.1f})'

        return True, 'ok'
    except Exception as e:
        logger.error(f'Quality check error: {e}')
        return True, 'ok'   # fail open — let later stages decide


# ── Stage 2 helper ────────────────────────────────────────────────────────────

def _detect_leaf(img: np.ndarray) -> tuple:
    """
    Detect whether an image contains a leaf using HSV colour analysis.
    Returns (is_leaf: bool, plant_ratio: float).
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = img.shape[0] * img.shape[1]

        # Green (healthy leaf): H 25-90
        green_mask = cv2.inRange(
            hsv,
            np.array([25, 25, 20]),
            np.array([90, 255, 255]),
        )

        # Yellow-green / early disease: H 15-25
        yellow_green_mask = cv2.inRange(
            hsv,
            np.array([15, 40, 20]),
            np.array([25, 255, 255]),
        )

        plant_mask = cv2.bitwise_or(green_mask, yellow_green_mask)
        plant_ratio = float(cv2.countNonZero(plant_mask)) / total_pixels

        return plant_ratio >= LEAF_COLOR_RATIO_MIN, plant_ratio

    except Exception as e:
        logger.error(f'Leaf detection error: {e}')
        return True, 1.0   # fail open


# ── Stage 3 helper ────────────────────────────────────────────────────────────

def _check_bean_leaf_shape(img: np.ndarray) -> tuple:
    """
    Validate that the dominant plant region has bean-leaf-like geometry.

    Bean leaves are broad, oval, and compound (trifoliate); they are NOT
    needle-like (grass, pine) or extremely fragmented.

    Returns (is_bean_like: bool, reason: str).
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Broad plant mask for shape analysis (wider hue range than Stage 2)
        plant_mask = cv2.inRange(
            hsv,
            np.array([12, 20, 15]),
            np.array([100, 255, 255]),
        )

        # Morphological clean-up to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean  = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        clean  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Colour passed Stage 2 but nothing contour-worthy — pass through
            return True, 'no_contours_pass'

        total_pixels = img.shape[0] * img.shape[1]
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        # Region must occupy at least 3 % of the image
        if area < total_pixels * 0.03:
            return False, 'leaf_region_too_small'

        # ── Solidity check ────────────────────────────────────────────────────
        hull      = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < BEAN_LEAF_MIN_SOLIDITY:
                return False, f'low_solidity_{solidity:.2f}'

        # ── Elongation check ─────────────────────────────────────────────────
        _, _, bw, bh = cv2.boundingRect(largest)
        elongation = max(bw, bh) / max(min(bw, bh), 1)
        if elongation > BEAN_LEAF_MAX_ELONGATION:
            return False, f'too_elongated_{elongation:.1f}x'

        return True, 'ok'

    except Exception as e:
        logger.error(f'Bean-leaf shape check error: {e}')
        return True, 'ok'   # fail open


# ── Image decoding utilities ──────────────────────────────────────────────────

def _get_raw_image(image_data) -> np.ndarray:
    """Decode image_data to a raw BGR numpy array (for Stages 1-3)."""
    try:
        if isinstance(image_data, bytes):
            arr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_data, str):
            return cv2.imread(str(image_data))
        if isinstance(image_data, np.ndarray):
            return image_data
        if isinstance(image_data, Image.Image):
            return cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f'Raw image decode error: {e}')
    return None


# ── Model class ───────────────────────────────────────────────────────────────

class AngularLeafSpotModel:
    """Service class for Angular Leaf Spot detection model."""

    def __init__(self, model_path="models/beenleaf_model.h5"):
        self.model_path  = model_path
        self.model       = None
        self.is_loaded   = False
        self._is_binary  = True   # set on first load based on output shape
        self._n_classes  = 2

    def load_model(self):
        """Load the trained model from disk."""
        if self.is_loaded:
            return True

        if not Path(self.model_path).exists():
            logger.error(f'Model not found at {self.model_path}')
            return False

        try:
            self.model = tf.keras.models.load_model(
                self.model_path, compile=False
            )
            # Auto-detect binary sigmoid vs multi-class softmax
            out_shape = self.model.output_shape
            n_out = out_shape[-1]
            self._is_binary = n_out == 1
            self._n_classes = 2 if self._is_binary else n_out
            self.is_loaded = True
            logger.info(
                f'Model loaded from {self.model_path} '
                f'({"binary sigmoid" if self._is_binary else f"{self._n_classes}-class softmax"})'
            )
            return True
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            return False

    def preprocess_image(self, image_data):
        """
        Preprocess image for ML inference.
        Returns numpy array of shape (1, 256, 256, 3) or None on failure.
        """
        try:
            if isinstance(image_data, str):
                if not Path(image_data).exists():
                    logger.error(f'Image file not found: {image_data}')
                    return None
                img = cv2.imread(str(image_data))
                if img is None:
                    logger.error(f'Could not load image: {image_data}')
                    return None

            elif isinstance(image_data, bytes):
                arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    logger.error('Could not decode image from bytes')
                    return None

            elif isinstance(image_data, Image.Image):
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            elif isinstance(image_data, np.ndarray):
                img = image_data

            else:
                logger.error(f'Unsupported image type: {type(image_data)}')
                return None

            resized    = tf.image.resize(img, (224, 224))
            normalized = resized / 255.0
            return np.expand_dims(normalized, 0)

        except Exception as e:
            logger.error(f'Preprocessing error: {e}')
            return None

    def predict(self, image_data):
        """
        Classify a leaf image through the four-stage pipeline.

        Returns a dict with keys:
          predicted_class : 'low_quality' | 'not_leaf' | 'not_bean_leaf'
                          | 'healthy' | 'angular_leaf_spot' | 'other_disease'
          status          : 'rejected' | 'not_leaf' | 'healthy' | 'unhealthy'
          health_status   : 'LOW_QUALITY' | 'NOT_LEAF' | 'NOT_BEAN_LEAF'
                          | 'HEALTHY' | 'UNHEALTHY'
          is_leaf         : bool
          confidence      : float (0-1)
          result          : human-readable label
          threshold       : boundary used
          interpretation  : explanation string
=======
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
            # Compatibility shim: older .h5 models store InputLayer config with
            # 'batch_shape', which newer Keras versions don't recognise.
            class _CompatInputLayer(tf.keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    if 'batch_shape' in kwargs:
                        kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
                    super().__init__(*args, **kwargs)

            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'InputLayer': _CompatInputLayer},
                compile=False,
            )
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
>>>>>>> c8c3b73 (feat: Implementing Angular leaf model)
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
<<<<<<< HEAD

        raw_img = _get_raw_image(image_data)

        if raw_img is not None:

            # ── Stage 1: Image quality ────────────────────────────────────────
            is_quality_ok, quality_reason = _check_image_quality(raw_img)
            if not is_quality_ok:
                logger.info(f'Stage 1 rejected: {quality_reason}')
                return {
                    'predicted_class': 'low_quality',
                    'status':          'rejected',
                    'health_status':   'LOW_QUALITY',
                    'is_leaf':         False,
                    'confidence':      0.0,
                    'result':          'Low Quality Image',
                    'threshold':       ALS_THRESHOLD,
                    'interpretation':  f'Image quality insufficient: {quality_reason}.',
                }

            # ── Stage 2: Leaf colour detection ────────────────────────────────
            is_leaf, plant_ratio = _detect_leaf(raw_img)
            if not is_leaf:
                logger.info(f'Stage 2 rejected: plant_ratio={plant_ratio:.3f}')
                return {
                    'predicted_class': 'not_leaf',
                    'status':          'not_leaf',
                    'health_status':   'NOT_LEAF',
                    'is_leaf':         False,
                    'confidence':      round(plant_ratio, 4),
                    'result':          'Not a Leaf',
                    'threshold':       ALS_THRESHOLD,
                    'interpretation':  'Image does not appear to contain a leaf.',
                }

            # ── Stage 3: Bean-leaf shape validation ───────────────────────────
            is_bean_like, shape_reason = _check_bean_leaf_shape(raw_img)
            if not is_bean_like:
                logger.info(f'Stage 3 rejected: {shape_reason}')
                return {
                    'predicted_class': 'not_bean_leaf',
                    'status':          'rejected',
                    'health_status':   'NOT_BEAN_LEAF',
                    'is_leaf':         True,   # a leaf was found, just not a bean leaf
                    'confidence':      0.0,
                    'result':          'Not a Bean Leaf',
                    'threshold':       ALS_THRESHOLD,
                    'interpretation':  (
                        f'A leaf was detected but its shape is not consistent '
                        f'with bean leaves ({shape_reason}). Only bean leaves are supported.'
                    ),
                }
        else:
            # Could not decode image — skip geometric stages, let ML try
            logger.warning('Could not decode raw image; skipping Stages 1-3.')

        # ── Stage 4: ML model inference ───────────────────────────────────────
        processed = self.preprocess_image(image_data)
        if processed is None:
            return None

        try:
            output = self.model.predict(processed, verbose=0)[0]

            if self._is_binary:
                return self._classify_binary(float(output[0]))
            else:
                return self._classify_multiclass(output)

        except Exception as e:
            logger.error(f'Prediction error: {e}')
            return None

    def _classify_binary(self, sigmoid: float) -> dict:
        """
        Classify using the original 2-class binary sigmoid model.
        Class 0 = healthy (low sigmoid), Class 1 = ALS (high sigmoid).
        """
        if sigmoid >= ALS_THRESHOLD:
            return {
                'predicted_class': 'angular_leaf_spot',
                'status':          'unhealthy',
                'health_status':   'UNHEALTHY',
                'is_leaf':         True,
                'confidence':      round(sigmoid, 4),
                'result':          'Angular Leaf Spot Detected',
                'threshold':       ALS_THRESHOLD,
                'interpretation':  f'sigmoid={sigmoid:.3f} ≥ {ALS_THRESHOLD} → ALS',
            }

        if sigmoid <= HEALTHY_THRESHOLD:
            confidence = round(1.0 - sigmoid, 4)
            return {
                'predicted_class': 'healthy',
                'status':          'healthy',
                'health_status':   'HEALTHY',
                'is_leaf':         True,
                'confidence':      confidence,
                'result':          'Healthy Leaf',
                'threshold':       ALS_THRESHOLD,
                'interpretation':  f'sigmoid={sigmoid:.3f} ≤ {HEALTHY_THRESHOLD} → healthy',
            }

        # Uncertainty band (HEALTHY_THRESHOLD < sigmoid < ALS_THRESHOLD)
        return {
            'predicted_class': 'other_disease',
            'status':          'unhealthy',
            'health_status':   'UNHEALTHY',
            'is_leaf':         True,
            'confidence':      round(0.5 + abs(sigmoid - 0.525), 4),
            'result':          'Other Leaf Condition Detected',
            'threshold':       ALS_THRESHOLD,
            'interpretation':  (
                f'sigmoid={sigmoid:.3f} between '
                f'{HEALTHY_THRESHOLD}–{ALS_THRESHOLD}'
                ' — uncertain, possibly another disease or stress condition.'
            ),
        }

    def _classify_multiclass(self, probs: np.ndarray) -> dict:
        """
        Classify using the 3-class softmax model trained with:
          Class 0 = angular_leaf_spot
          Class 1 = healthy
          Class 2 = other_leaves

        Folder names are sorted alphabetically by Keras, so:
          data/angular_leaf_spot/ → index 0
          data/healthy/           → index 1
          data/other_leaves/      → index 2
        """
        CLASS_ALS        = 0
        CLASS_HEALTHY    = 1
        CLASS_OTHER_LEAF = 2

        predicted_idx  = int(np.argmax(probs))
        confidence     = float(probs[predicted_idx])

        # Reject if the model is very uncertain overall
        MIN_CONFIDENCE = 0.50
        if confidence < MIN_CONFIDENCE:
            return {
                'predicted_class': 'other_disease',
                'status':          'unhealthy',
                'health_status':   'UNHEALTHY',
                'is_leaf':         True,
                'confidence':      round(confidence, 4),
                'result':          'Uncertain Classification',
                'threshold':       MIN_CONFIDENCE,
                'interpretation':  (
                    f'Max softmax probability {confidence:.3f} < {MIN_CONFIDENCE}'
                    ' — model cannot confidently classify this leaf.'
                ),
            }

        if predicted_idx == CLASS_ALS:
            return {
                'predicted_class': 'angular_leaf_spot',
                'status':          'unhealthy',
                'health_status':   'UNHEALTHY',
                'is_leaf':         True,
                'confidence':      round(confidence, 4),
                'result':          'Angular Leaf Spot Detected',
                'threshold':       MIN_CONFIDENCE,
                'interpretation':  f'softmax[als]={confidence:.3f} → ALS',
            }

        if predicted_idx == CLASS_HEALTHY:
            return {
                'predicted_class': 'healthy',
                'status':          'healthy',
                'health_status':   'HEALTHY',
                'is_leaf':         True,
                'confidence':      round(confidence, 4),
                'result':          'Healthy Leaf',
                'threshold':       MIN_CONFIDENCE,
                'interpretation':  f'softmax[healthy]={confidence:.3f} → healthy',
            }

        # CLASS_OTHER_LEAF (and any unexpected extra class)
        return {
            'predicted_class': 'not_bean_leaf',
            'status':          'rejected',
            'health_status':   'NOT_BEAN_LEAF',
            'is_leaf':         True,
            'confidence':      round(confidence, 4),
            'result':          'Not a Bean Leaf',
            'threshold':       MIN_CONFIDENCE,
            'interpretation':  (
                f'softmax[other_leaves]={confidence:.3f}'
                ' — model identifies this as a non-bean leaf.'
            ),
        }

    def predict_batch(self, image_data_list):
        """Predict a list of images and return a list of result dicts."""
        if not self.is_loaded:
            if not self.load_model():
                return None

=======
        
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
        
>>>>>>> c8c3b73 (feat: Implementing Angular leaf model)
        results = []
        for i, image_data in enumerate(image_data_list):
            result = self.predict(image_data)
            if result:
                result['image_index'] = i
                results.append(result)
            else:
<<<<<<< HEAD
                results.append({'image_index': i, 'error': 'Failed to process image'})

        return results


# Global singleton used by the API server
model_service = AngularLeafSpotModel()
=======
                results.append({
                    'image_index': i,
                    'error': 'Failed to process image'
                })
        
        return results

# Global model instance for the API
model_service = AngularLeafSpotModel()
>>>>>>> c8c3b73 (feat: Implementing Angular leaf model)
