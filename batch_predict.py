#!/usr/bin/env python3
"""
Batch prediction script for Angular Leaf Spot Detection
Tests multiple images at once
"""

import os
import glob
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

def load_model():
    """Load the trained model"""
    model_path = "models/beenleaf_model.h5"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_batch_images(model, image_paths):
    """Predict multiple images"""
    results = []
    
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"⚠️ Skipping missing image: {image_path}")
            continue
            
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ Could not load image: {image_path}")
                continue
                
            # Resize and normalize
            resized = tf.image.resize(img, (256, 256))
            normalized = resized / 255.0
            input_batch = np.expand_dims(normalized, 0)
            
            # Make prediction
            prediction = model.predict(input_batch, verbose=0)
            confidence = float(prediction[0][0])
            
            # Interpret result
            if confidence > 0.5:
                status = "UNHEALTHY"
                emoji = "🔴"
            else:
                status = "HEALTHY"
                emoji = "🟢"
                
            results.append({
                'image': os.path.basename(image_path),
                'status': status,
                'confidence': confidence,
                'emoji': emoji
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    return results

def main():
    print("🌿 Angular Leaf Spot Detection - Batch Testing")
    print("=" * 50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Find all image files in current directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(ext))
        all_images.extend(glob.glob(ext.upper()))
    
    if not all_images:
        print("❌ No image files found in current directory")
        return
    
    # Remove any model files from the list
    image_files = [img for img in all_images if not img.startswith('models/')]
    
    print(f"📸 Found {len(image_files)} images to test")
    print("-" * 50)
    
    # Predict all images
    results = predict_batch_images(model, image_files)
    
    if not results:
        print("❌ No valid predictions made")
        return
    
    # Display results
    print("\n📊 BATCH PREDICTION RESULTS:")
    print("=" * 70)
    print(f"{'Image':<25} {'Status':<12} {'Confidence':<12} {'Result'}")
    print("-" * 70)
    
    healthy_count = 0
    unhealthy_count = 0
    
    for result in results:
        print(f"{result['image']:<25} {result['status']:<12} {result['confidence']:<12.4f} {result['emoji']}")
        
        if result['status'] == 'HEALTHY':
            healthy_count += 1
        else:
            unhealthy_count += 1
    
    print("-" * 70)
    print(f"📈 SUMMARY:")
    print(f"   🟢 Healthy leaves: {healthy_count}")
    print(f"   🔴 Unhealthy leaves (Angular Leaf Spot): {unhealthy_count}")
    print(f"   📊 Total tested: {len(results)}")
    
    if unhealthy_count > 0:
        percentage = (unhealthy_count / len(results)) * 100
        print(f"   ⚠️  Disease detection rate: {percentage:.1f}%")
    
    print(f"\n✅ Batch prediction completed!")

if __name__ == "__main__":
    main()