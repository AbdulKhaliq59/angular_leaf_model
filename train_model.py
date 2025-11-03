#!/usr/bin/env python3
"""
Complete Angular Leaf Spot Detection Training Script
Runs the full training pipeline and saves the model
"""

import tensorflow as tf
import os
import cv2 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

def setup_gpu():
    """Setup GPU if available"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU setup complete. Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    else:
        print("ℹ️ No GPU found, using CPU")

def clean_data(data_dir='data'):
    """Remove corrupted images"""
    print("🧹 Cleaning data...")
    image_exts = ['.jpeg', '.jpg', '.png', '.bmp']
    removed_count = 0
    
    for image_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, image_class)
        if not os.path.isdir(class_path):
            continue
            
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            try:
                # Check if file extension is valid
                file_ext = os.path.splitext(image.lower())[1]
                if file_ext not in image_exts:
                    print(f'Removing bad image: {image_path}')
                    os.remove(image_path)
                    removed_count += 1
                else:
                    # Try to load the image to verify it's valid
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f'Removing corrupted image: {image_path}')
                        os.remove(image_path)
                        removed_count += 1
            except Exception as e:
                print(f'Error with image {image_path}: {e}')
                
    print(f"✅ Data cleaning complete. Removed {removed_count} bad images")

def load_and_prepare_data():
    """Load and preprocess the data"""
    print("📊 Loading data...")
    
    # Load dataset
    data = tf.keras.utils.image_dataset_from_directory('data')
    class_names = data.class_names
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    
    print(f"✅ Data loaded. Classes: {class_names}")
    print(f"   - Class 0: {class_names[0]}")
    print(f"   - Class 1: {class_names[1]}")
    
    # Scale the data
    data = data.map(lambda x, y: (x / 255, y))
    
    # Split the data
    data_size = len(data)
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = data_size - train_size - val_size
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    print(f"📊 Data split:")
    print(f"   - Training: {train_size} batches")
    print(f"   - Validation: {val_size} batches")
    print(f"   - Test: {test_size} batches")
    
    return train, val, test, class_names

def create_model():
    """Create the CNN model"""
    print("🏗️ Creating model...")
    
    model = Sequential([
        Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), 1, activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), 1, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    print("✅ Model created!")
    print(f"   - Total parameters: {model.count_params():,}")
    
    return model

def train_model(model, train, val, epochs=20):
    """Train the model"""
    print(f"🚀 Starting training for {epochs} epochs...")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Set up TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    
    # Train the model
    history = model.fit(
        train, 
        epochs=epochs, 
        validation_data=val, 
        callbacks=[tensorboard_callback],
        verbose=1
    )
    
    print("✅ Training completed!")
    return history

def evaluate_model(model, test):
    """Evaluate the model"""
    print("📈 Evaluating model...")
    
    from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
    
    pre = Precision()
    rec = Recall()
    acc = BinaryAccuracy()
    
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X, verbose=0)
        pre.update_state(y, yhat)
        rec.update_state(y, yhat)
        acc.update_state(y, yhat)
    
    precision = pre.result().numpy()
    recall = rec.result().numpy()
    accuracy = acc.result().numpy()
    
    print(f"📊 Final Results:")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - Accuracy: {accuracy:.4f}")
    
    return precision, recall, accuracy

def save_model(model, model_path='models/beenleaf_model.h5'):
    """Save the trained model"""
    print("💾 Saving model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save(model_path)
    
    print(f"✅ Model saved to {model_path}")
    return model_path

def test_prediction(model, test_image='healthy1.jpeg'):
    """Test the model with a sample image"""
    if not os.path.exists(test_image):
        print(f"⚠️ Test image {test_image} not found")
        return
        
    print(f"🔍 Testing with {test_image}...")
    
    # Load and preprocess image
    img = cv2.imread(test_image)
    resized = tf.image.resize(img, (256, 256))
    normalized = resized / 255.0
    input_batch = np.expand_dims(normalized, 0)
    
    # Make prediction
    prediction = model.predict(input_batch, verbose=0)
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        result = "🔴 UNHEALTHY (Angular Leaf Spot)"
    else:
        result = "🟢 HEALTHY"
    
    print(f"   Result: {result}")
    print(f"   Confidence: {confidence:.4f}")

def main():
    print("🌿 Angular Leaf Spot Detection - Training Pipeline")
    print("=" * 50)
    
    # Setup
    setup_gpu()
    clean_data()
    
    # Load data
    train, val, test, class_names = load_and_prepare_data()
    
    # Create and train model
    model = create_model()
    history = train_model(model, train, val, epochs=20)
    
    # Evaluate
    evaluate_model(model, test)
    
    # Save model
    model_path = save_model(model)
    
    # Test prediction
    test_prediction(model)
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"Model saved at: {model_path}")
    print("You can now use predict_single_image.py to test new images.")

if __name__ == "__main__":
    main()