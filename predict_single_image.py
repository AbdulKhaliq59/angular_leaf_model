#!/usr/bin/env python3
"""
Quick prediction script for Angular Leaf Spot Detection
Usage: python predict_single_image.py <image_path>
"""

import sys
from pathlib import Path
from model_service import model_service
        
def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_single_image.py <image_path>")
        print("\nAvailable test images:")
        test_images = ["healthy1.jpeg", "health_bean.jpeg", "angula_leaf_spot.jpeg", "healthy.jpeg"]
        for img in test_images:
            if Path(img).exists():
                print(f"  - {img}")
        return

    image_path = sys.argv[1]

    print("🌿 Angular Leaf Spot Detection System")
    print("=" * 40)

    # Load model
    if not model_service.load_model():
        return

    # Make prediction
    result = model_service.predict(image_path)
    if result:
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"Image: {image_path}")
        print(f"Status: 🔴 {result['health_status']}" if result['status'] == 'unhealthy' else f"Status: 🟢 {result['health_status']}")
        print(f"Confidence Score: {result['confidence']:.4f}")
        print(f"Threshold: {result['threshold']} (>{result['threshold']} = unhealthy, ≤{result['threshold']} = healthy)")
        print(f"\n✅ Prediction completed successfully!")
    else:
        print("❌ Prediction failed!")

if __name__ == "__main__":
    main()