#!/usr/bin/env python3
"""
Angular Leaf Spot Detection — Multi-class Training Pipeline
Uses MobileNetV2 transfer learning with augmentation and class-balanced training.

Supported folder layouts inside data/:

  2-class (binary — original):
    data/angular_leaf_spot/
    data/healthy/

  3-class (recommended — rejects other-leaf images like avocado, mango …):
    data/angular_leaf_spot/
    data/healthy/
    data/other_leaves/        ← add images of non-bean leaves here

The script auto-detects the number of classes and switches between a sigmoid
output (2-class) and a softmax output (3+ classes) accordingly.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224          # MobileNetV2 expects 224×224
BATCH_SIZE   = 16
EPOCHS_HEAD  = 15           # train only the new head first
EPOCHS_FINE  = 25           # then fine-tune the top layers of the base
DATA_DIR     = 'data'
MODEL_PATH   = 'models/beenleaf_model.h5'
SEED         = 42


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ready ({len(gpus)} device(s))")
    else:
        print("ℹ️  No GPU found, using CPU")


# ── Data loading ──────────────────────────────────────────────────────────────

def count_classes(data_dir: str) -> int:
    return sum(
        1 for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )


def load_datasets():
    """
    Load train / val splits from data/.
    Auto-detects binary (2-class) vs multi-class based on folder count.
    Applies strong augmentation to the training set only.
    """
    n_classes = count_classes(DATA_DIR)
    is_binary = n_classes == 2
    label_mode = 'binary' if is_binary else 'int'

    print(f"\n📂 Detected {n_classes} class folder(s) → mode='{label_mode}'")

    # Raw datasets (no augmentation — needed for class-weight calculation)
    train_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=label_mode,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=label_mode,
    )

    class_names = train_raw.class_names
    print(f"   Classes: {class_names}")
    for i, name in enumerate(class_names):
        arrow = 'sigmoid ≈ 0' if (is_binary and i == 0) else ('sigmoid ≈ 1' if is_binary else f'softmax[{i}]')
        print(f"   Class {i} = '{name}' → {arrow}")
    print()

    # Count samples per class for class-weight calculation
    label_counts = {i: 0 for i in range(n_classes)}
    for _, labels in train_raw:
        for lbl in labels.numpy().ravel():
            label_counts[int(lbl)] += 1
    total = sum(label_counts.values())
    class_weight = {
        cls: total / (n_classes * max(count, 1))
        for cls, count in label_counts.items()
    }
    for cls, count in label_counts.items():
        print(f"📊 Class {cls} '{class_names[cls]}': {count} samples  (weight={class_weight[cls]:.3f})")
    print()

    # Augmentation layer (GPU-accelerated, applied only during training)
    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name='augmentation')

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def prepare_train(image, label):
        image = augment(image, training=True)
        image = preprocess(image)
        return image, label

    def prepare_eval(image, label):
        image = preprocess(image)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_raw
        .map(prepare_train, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(500)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw
        .map(prepare_eval, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names, class_weight, n_classes


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int):
    """
    MobileNetV2 base (frozen) + custom classification head.
    Output: sigmoid (binary) or softmax (multi-class).
    Two-stage training: head only → fine-tune last 30 layers.
    """
    is_binary = n_classes == 2

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    if is_binary:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss    = 'binary_crossentropy'
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        loss    = 'sparse_categorical_crossentropy'

    # AUC only works reliably with binary sigmoid; use accuracy for multi-class
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')] if is_binary else ['accuracy']
    monitor = 'val_auc' if is_binary else 'val_accuracy'

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=metrics,
    )
    print(f"✅ Model built ({'binary sigmoid' if is_binary else f'{n_classes}-class softmax'}) "
          f"— trainable params: {model.count_params():,}")
    return model, base, loss, metrics, monitor


def callbacks_for(model_path: str, monitor: str):
    return [
        ModelCheckpoint(
            model_path,
            monitor=monitor,
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train(model, base, loss, metrics, monitor, train_ds, val_ds, class_weight):
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Stage 1: train head only ──────────────────────────────────────────────
    print("\n🚀 Stage 1 — Training head (base frozen) …")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=callbacks_for(MODEL_PATH, monitor),
        verbose=1,
    )

    # ── Stage 2: fine-tune top layers of MobileNetV2 ─────────────────────────
    print("\n🔓 Stage 2 — Fine-tuning top 30 layers of base …")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=loss,
        metrics=metrics,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight,
        callbacks=callbacks_for(MODEL_PATH, monitor),
        verbose=1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, val_ds):
    print("\n📈 Final evaluation on validation set …")
    results = model.evaluate(val_ds, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"   {name}: {val:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🌿 Angular Leaf Spot Detection — Multi-class Training Pipeline")
    print("=" * 62)

    setup_gpu()

    if not os.path.isdir(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found.")
        print("   Create sub-folders for each class, e.g.:")
        print(f"   {DATA_DIR}/angular_leaf_spot/  — diseased bean leaves")
        print(f"   {DATA_DIR}/healthy/             — healthy bean leaves")
        print(f"   {DATA_DIR}/other_leaves/        — avocado, mango, etc.  (optional but recommended)")
        return

    train_ds, val_ds, class_names, class_weight, n_classes = load_datasets()

    if n_classes < 2:
        print("❌ Need at least 2 class folders inside data/.")
        return

    if n_classes == 2:
        print("⚠️  Running in 2-class (binary) mode.")
        print("   To reject non-bean leaves (avocado etc.), add a data/other_leaves/ folder")
        print("   containing images of various non-bean leaves and re-run.\n")
    else:
        print(f"✅ Running in {n_classes}-class mode — non-bean leaves will be explicitly trained.\n")

    model, base, loss, metrics, monitor = build_model(n_classes)
    train(model, base, loss, metrics, monitor, train_ds, val_ds, class_weight)

    print(f"\n💾 Best model saved to {MODEL_PATH}")
    evaluate(model, val_ds)

    print("\n🎉 Training complete!")
    print(f"   Restart the Flask server to load the new model.")
    print(f"\n📋 Class index → folder mapping:")
    for i, name in enumerate(class_names):
        print(f"   {i} → {name}")
    print("   Update ALS_THRESHOLD / HEALTHY_THRESHOLD in model_service.py after evaluating the new model.")


if __name__ == '__main__':
    main()
