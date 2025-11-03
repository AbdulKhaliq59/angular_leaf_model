#!/usr/bin/env python3
"""
Test script for Angular Leaf Spot Detection API
Tests various endpoints to ensure the API is working correctly
"""

import requests
import json
import os
import sys
from pathlib import Path

def test_health_check(base_url):
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Model loaded: {data.get('model_loaded', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info(base_url):
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved successfully")
            print(f"   Model path: {data.get('model_path')}")
            print(f"   Input size: {data.get('input_size')}")
            print(f"   Classes: {data.get('classes')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_predict_endpoint(base_url, image_path=None):
    """Test the prediction endpoint"""
    print("\n🔍 Testing prediction endpoint...")
    
    # Try to find a test image
    if image_path and Path(image_path).exists():
        test_image = image_path
    else:
        # Look for common test image names
        possible_images = [
            "test_image.jpg", "test.jpg", "sample.jpg", "image.jpg",
            "data/healthy/healthy1.jpg", "data/angular_leaf_spot/spot1.jpg"
        ]
        
        test_image = None
        for img in possible_images:
            if Path(img).exists():
                test_image = img
                break
    
    if not test_image:
        print("⚠️ No test image found. Skipping prediction test.")
        print("   To test predictions, provide an image file.")
        return True
    
    try:
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data['prediction']
                print(f"✅ Prediction successful")
                print(f"   Status: {prediction.get('health_status')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.4f}")
                print(f"   Result: {prediction.get('result')}")
                return True
            else:
                print(f"❌ Prediction failed: {data}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_url_prediction(base_url):
    """Test the URL prediction endpoint"""
    print("\n🔍 Testing URL prediction endpoint...")
    
    # Use a sample image URL (you can replace with a real image URL)
    test_url = "https://via.placeholder.com/256x256/00FF00/FFFFFF?text=Test+Leaf"
    
    try:
        data = {'url': test_url}
        response = requests.post(f"{base_url}/predict/url", json=data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ URL prediction successful")
                prediction = result['prediction']
                print(f"   Status: {prediction.get('health_status')}")
                print(f"   Source URL: {prediction.get('source_url')}")
                return True
            else:
                print(f"❌ URL prediction failed: {result}")
                return False
        else:
            print(f"❌ URL prediction request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ URL prediction error: {e}")
        return False

def main():
    """Main test function"""
    # Get base URL from command line or use default
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = "http://localhost:5000"
    
    # Get optional image path
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🧪 Angular Leaf Spot Detection API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {base_url}")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test health check
    total_tests += 1
    if test_health_check(base_url):
        tests_passed += 1
    
    # Test model info
    total_tests += 1
    if test_model_info(base_url):
        tests_passed += 1
    
    # Test prediction
    total_tests += 1
    if test_predict_endpoint(base_url, image_path):
        tests_passed += 1
    
    # Test URL prediction (optional)
    total_tests += 1
    if test_url_prediction(base_url):
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the API server and model.")
        sys.exit(1)

if __name__ == "__main__":
    main()