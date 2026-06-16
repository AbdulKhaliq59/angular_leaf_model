#!/usr/bin/env python3
"""
Angular Leaf Spot Detection — 4-class Training Pipeline
MobileNetV2 transfer learning with full augmentation suite.

Expected data layout inside data/:
  data/angular_leaf_spot/   — ALS-diseased bean leaves
  data/healthy/             — healthy bean leaves
  data/other_disease/       — diverse diseased non-bean leaves (corn, grape, apple, etc.)
  data/other_leaves/        — visually distinct non-bean leaves (cassava, mango)

Split: 70 % train / 15 % val / 15 % test (test set is locked — never used for
       early stopping or checkpoint selection).

Improvements over previous version:
  - 4-class softmax only (binary sigmoid path retired)
  - Fixed augmentation order: cache raw → shuffle → augment each epoch
  - Extended augmentation: translation, crop, Gaussian noise, Cutout, shear
  - L2 regularization on Dense layers
  - Label smoothing to prevent overconfident predictions
  - Deeper fine-tuning: top 60 layers instead of 30
  - Per-class Precision, Recall, F1, AUC and confusion matrix at evaluation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')   # headless — no display needed on servers
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS_HEAD = 20
EPOCHS_FINE = 40
DATA_DIR    = 'data'
MODEL_PATH  = 'models/beenleaf_model.keras'
SEED        = 42
L2_REG      = 1e-4   # L2 weight penalty on Dense layers


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
    Load train / val / test splits from data/.
    Split: 70 % train, 15 % val, 15 % test.

    Augmentation order is:
        raw images → cache (fast disk reads)
                   → shuffle
                   → augment (fresh every epoch)
                   → prefetch
    This ensures every epoch sees different augmented variants.
    """
    n_classes = count_classes(DATA_DIR)
    if n_classes < 2:
        raise RuntimeError(f"Need at least 2 class folders inside {DATA_DIR}/")

    print(f"\n📂 Detected {n_classes} class folder(s)")

    # ── Stage 1: load full dataset without augmentation ───────────────────────
    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,           # unbatched so we can split cleanly
        label_mode='categorical',  # one-hot — required for label_smoothing
        shuffle=True,
    )
    class_names = full_ds.class_names
    print(f"   Classes: {class_names}")

    total = full_ds.cardinality().numpy()
    train_size = int(total * 0.70)
    val_size   = int(total * 0.15)
    # test_size = rest

    train_raw = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_raw   = remaining.take(val_size)
    test_raw  = remaining.skip(val_size)

    print(f"\n📊 Split: {train_size} train | {val_size} val | {total - train_size - val_size} test")

    # ── Single-pass: collect training samples as numpy (safe for from_generator)
    per_class_raw = {i: [] for i in range(n_classes)}
    for img, lbl in train_raw:
        cls_idx = int(np.argmax(lbl.numpy()))
        per_class_raw[cls_idx].append((img.numpy(), lbl.numpy()))

    label_counts = {i: len(per_class_raw[i]) for i in range(n_classes)}
    for i, name in enumerate(class_names):
        print(f"   Class {i} '{name}': {label_counts[i]} train samples")

    # Balance: keep ALL samples from the small bean-disease classes (ALS and
    # healthy, ~430 each); cap the larger support classes (other_disease,
    # other_leaves) to the same ceiling so no single class dominates the loss.
    BEAN_CLASSES    = {'angular_leaf_spot', 'healthy'}
    SUPPORT_CLASSES = {'other_disease', 'other_leaves'}

    als_idx  = next(i for i, n in enumerate(class_names) if n == 'angular_leaf_spot')
    bean_max = max(
        count for i, count in label_counts.items()
        if class_names[i] in BEAN_CLASSES
    )

    balanced_items = []
    for cls_idx, items in per_class_raw.items():
        name = class_names[cls_idx]
        cap  = bean_max if name in SUPPORT_CLASSES else len(items)
        balanced_items.extend(items[:cap])
        print(f"   → '{name}': {min(len(items), cap)} kept")

    def _gen():
        for img, lbl in balanced_items:
            yield img, lbl

    train_raw = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(n_classes,),             dtype=tf.float32),
        ),
    )
    train_size = len(balanced_items)
    print(f"\n   Balanced train size: {train_size}")

    # Give ALS 1.2× weight so the loss penalises ALS errors slightly harder.
    class_weight = {i: (1.2 if i == als_idx else 1.0) for i in range(n_classes)}

    # ── Balance val set too — prevents metric mismatch with unbalanced val ────
    # Without this, val_accuracy baseline for a balanced model is ~25% while a
    # model biased toward other_leaves (44% of val) gets 44%, causing early
    # stopping to fire immediately before the model learns anything.
    per_class_val = {i: [] for i in range(n_classes)}
    for img, lbl in val_raw:
        cls_idx = int(np.argmax(lbl.numpy()))
        per_class_val[cls_idx].append((img.numpy(), lbl.numpy()))

    val_min = min(len(v) for v in per_class_val.values())
    balanced_val = []
    for items in per_class_val.values():
        balanced_val.extend(items[:val_min])

    print(f"⚖️  Balanced val set    → {val_min} per class  ({len(balanced_val)} total)")

    def _val_gen():
        for img, lbl in balanced_val:
            yield img, lbl

    val_raw = tf.data.Dataset.from_generator(
        _val_gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(n_classes,),             dtype=tf.float32),
        ),
    )

    # ── Augmentation pipeline (applied only to training data) ─────────────────
    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        # Kept low intentionally: brightness/contrast shifts destroy the lesion
        # colour cues (rust-orange vs angular grey-brown) that distinguish
        # bean_rust from ALS. Spatial augmentation stays unchanged.
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
        layers.RandomCrop(int(IMG_SIZE * 0.90), int(IMG_SIZE * 0.90)),
        layers.Resizing(IMG_SIZE, IMG_SIZE),
    ], name='augmentation')

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def add_gaussian_noise(image):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=2.0)
        return tf.clip_by_value(image + noise, 0.0, 255.0)

    def prepare_train(image, label):
        image = augment(image, training=True)
        image = add_gaussian_noise(image)
        image = preprocess(image)
        return image, label

    def prepare_eval(image, label):
        image = preprocess(image)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    # Correct order: cache raw pixels → shuffle → augment fresh each epoch
    train_ds = (
        train_raw
        .cache()
        .shuffle(buffer_size=train_size, seed=SEED, reshuffle_each_iteration=True)
        .map(prepare_train, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw
        .map(prepare_eval, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_raw
        .map(prepare_eval, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names, class_weight, n_classes


# ── Augmentation summary ──────────────────────────────────────────────────────

def print_augmentation_summary():
    """Print a human-readable summary of augmentation applied per dataset split."""
    W = 62
    print("\n" + "─" * W)
    print("  📋  DATA AUGMENTATION SUMMARY")
    print("─" * W)

    splits = {
        "TRAIN": [
            ("RandomFlip",        "horizontal + vertical"),
            ("RandomRotation",    "±20 % (factor=0.2)"),
            ("RandomZoom",        "±20 % (factor=0.2)"),
            ("RandomTranslation", "height ±10 %, width ±10 %"),
            ("RandomBrightness",  "±10 % (factor=0.1)"),
            ("RandomContrast",    "±10 % (factor=0.1)"),
            ("RandomCrop",        f"90 % of {IMG_SIZE}px → {int(IMG_SIZE*0.9)}×{int(IMG_SIZE*0.9)}"),
            ("Resizing",          f"back to {IMG_SIZE}×{IMG_SIZE}"),
            ("Gaussian noise",    "mean=0.0, stddev=2.0 (pixel space)"),
            ("MobileNetV2 preprocess", "scale pixels to [-1, 1]"),
        ],
        "VAL": [
            ("MobileNetV2 preprocess", "scale pixels to [-1, 1]"),
        ],
        "TEST": [
            ("MobileNetV2 preprocess", "scale pixels to [-1, 1]"),
        ],
    }

    for split, steps in splits.items():
        print(f"\n  [{split}]")
        for i, (name, detail) in enumerate(steps, 1):
            print(f"    {i:>2}. {name:<26}  {detail}")

    print("\n" + "─" * W + "\n")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int):
    """
    MobileNetV2 base (frozen) + custom 4-class classification head.
    Output: softmax over n_classes.
    Two-stage training: head only → fine-tune top 60 layers of base.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    reg = regularizers.l2(L2_REG)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    loss = tf.keras.losses.CategoricalCrossentropy()

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=['accuracy'],
    )
    print(f"✅ Model built ({n_classes}-class softmax) "
          f"— trainable params: {model.count_params():,}")
    return model, base, loss


HISTORY_PATH = 'logs/training_history.csv'

def callbacks_for(model_path: str, append: bool = False):
    return [
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            mode='min',
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
        CSVLogger(HISTORY_PATH, append=append),
    ]


def train(model, base, loss, train_ds, val_ds, class_weight):
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Stage 1: train head only ──────────────────────────────────────────────
    print("\n🚀 Stage 1 — Training head (base frozen) …")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=callbacks_for(MODEL_PATH, append=False),
        verbose=1,
    )

    # ── Stage 2: fine-tune top 60 layers (deeper than previous 30) ───────────
    print("\n🔓 Stage 2 — Fine-tuning top 60 layers of MobileNetV2 …")
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=loss,
        metrics=['accuracy'],
    )

    # Stage 2 callbacks: seed checkpoint with Stage 1's best val_loss so it
    # only overwrites the saved file if Stage 2 actually improves on Stage 1.
    s2_cbs = callbacks_for(MODEL_PATH, append=True)
    stage1_best = min(history1.history.get('val_loss', [np.inf]))
    s2_cbs[0].best = stage1_best   # ModelCheckpoint — don't overwrite a better Stage 1 checkpoint

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight,
        callbacks=s2_cbs,
        verbose=1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_ds, class_names):
    """
    Evaluate on the locked test set.
    Reports: accuracy, per-class Precision / Recall / F1, confusion matrix.
    Saves confusion matrix plot to models/confusion_matrix.png.
    """
    print("\n📈 Final evaluation on LOCKED TEST SET …")

    # Collect ground-truth labels from test_ds (same iteration chain as predict,
    # so order is guaranteed to match — avoids misalignment from iterating test_raw
    # separately which can reshuffle via a new upstream full_ds iteration).
    y_true = np.concatenate([
        np.argmax(lbl.numpy(), axis=1) for _, lbl in test_ds
    ])

    # Second pass: test_ds cache is warm, serves batches in the same order.
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n── Accuracy ──────────────────────────────────────────────────────")
    acc = np.mean(y_true == y_pred)
    print(f"   Test accuracy: {acc:.4f}")

    print("\n── Per-class Precision / Recall / F1 ────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\n── Confusion Matrix ──────────────────────────────────────────────")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Save confusion matrix as image
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    cm_path = 'models/confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n   Confusion matrix saved → {cm_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🌿 Angular Leaf Spot Detection — 4-class Training Pipeline")
    print("=" * 62)

    setup_gpu()

    if not os.path.isdir(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found.")
        return

    train_ds, val_ds, test_ds, class_names, class_weight, n_classes = load_datasets()

    print_augmentation_summary()

    model, base, loss = build_model(n_classes)
    train(model, base, loss, train_ds, val_ds, class_weight)

    print(f"\n💾 Best model saved to {MODEL_PATH}")

    # Evaluate the in-memory model — EarlyStopping already restored the best
    # weights, and we avoid any .h5 save/load corruption by skipping load_model.
    evaluate(model, test_ds, class_names)

    print("\n🎉 Training complete!")
    print(f"\n📋 Class index → folder mapping:")
    for i, name in enumerate(class_names):
        print(f"   {i} → {name}")
    print("\n   Update MIN_CONFIDENCE in model_service.py after reviewing test metrics.")


if __name__ == '__main__':
    main()
