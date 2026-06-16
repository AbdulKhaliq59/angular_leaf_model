#!/usr/bin/env python3
"""
Model Performance Scoreboard
Evaluates the trained MobileNetV2 model on the held-out test split and
produces a publication-quality bar chart of Accuracy, Precision, Recall,
and F1-Score — both per class and overall (macro average).

Run from the angular_leaf_model/ directory:
    python plot_performance.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Config (must match train_model.py) ────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
DATA_DIR   = 'data'
MODEL_PATH = 'models/beenleaf_model.keras'
SEED       = 42
OUT_PATH   = 'logs/performance_scoreboard.png'

CLASS_DISPLAY = {
    'angular_leaf_spot': 'Angular\nLeaf Spot',
    'healthy':           'Healthy',
    'other_disease':     'Other\nDisease',
    'other_leaves':      'Other\nLeaves',
}

# Colour palette: one per class + one for the overall bar
PALETTE = {
    'angular_leaf_spot': '#D32F2F',   # red    — disease alert
    'healthy':           '#388E3C',   # green  — healthy
    'other_disease':     '#F57C00',   # orange — other disease
    'other_leaves':      '#1976D2',   # blue   — non-bean
    'overall':           '#6A1B9A',   # purple — macro avg
}

METRIC_COLORS = ['#1565C0', '#2E7D32', '#E65100', '#4A148C']  # P, R, F1, Acc


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_split():
    """Reproduce the exact same 70/15/15 split used during training."""
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def prepare_eval(image, label):
        # Must match train_model.py's prepare_eval: scale [0,255] → [-1,1]
        return preprocess(image), label

    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        label_mode='categorical',
        shuffle=True,
    )
    class_names = full_ds.class_names
    total      = full_ds.cardinality().numpy()
    train_size = int(total * 0.70)
    val_size   = int(total * 0.15)

    test_ds = (
        full_ds
        .skip(train_size + val_size)
        .map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"  Test samples : {total - train_size - val_size}")
    print(f"  Classes      : {class_names}")
    return test_ds, class_names


def evaluate(model, test_ds, class_names):
    """Run inference and return true labels, predicted labels, and probabilities."""
    y_true, y_pred_prob = [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_prob.extend(probs)

    y_true      = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    return y_true, y_pred


def compute_metrics(y_true, y_pred, class_names):
    """Return per-class and macro-avg dicts with P, R, F1, Accuracy."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    per_class = {}
    for name in class_names:
        per_class[name] = {
            'Precision': report[name]['precision'],
            'Recall':    report[name]['recall'],
            'F1-Score':  report[name]['f1-score'],
        }

    overall = {
        'Precision': report['macro avg']['precision'],
        'Recall':    report['macro avg']['recall'],
        'F1-Score':  report['macro avg']['f1-score'],
        'Accuracy':  accuracy,
    }
    return per_class, overall


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_scoreboard(per_class, overall, class_names):
    os.makedirs('logs', exist_ok=True)

    metrics      = ['Precision', 'Recall', 'F1-Score']
    n_classes    = len(class_names)
    bar_w        = 0.22
    x_class      = np.arange(n_classes)

    fig = plt.figure(figsize=(16, 10), facecolor='#0F172A')
    gs  = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35,
                   left=0.07, right=0.97, top=0.88, bottom=0.10)

    title_kw   = dict(color='white', fontsize=11, fontweight='bold', pad=10)
    label_kw   = dict(color='#CBD5E1', fontsize=9)
    tick_kw    = dict(colors='#94A3B8', labelsize=8.5)

    # ── Panel 1: per-class grouped bars (P, R, F1) ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1E293B')

    for m_idx, (metric, color) in enumerate(zip(metrics, METRIC_COLORS)):
        offsets = x_class + (m_idx - 1) * bar_w
        values  = [per_class[c][metric] for c in class_names]
        bars    = ax1.bar(offsets, values, bar_w, color=color, alpha=0.88,
                          label=metric, zorder=3)
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{val:.2f}',
                ha='center', va='bottom',
                color='white', fontsize=7.5, fontweight='bold',
            )

    ax1.set_title('Per-Class Precision · Recall · F1-Score', **title_kw)
    ax1.set_xticks(x_class)
    ax1.set_xticklabels(
        [CLASS_DISPLAY.get(c, c) for c in class_names],
        color='#CBD5E1', fontsize=9.5,
    )
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel('Score', **label_kw)
    ax1.tick_params(axis='y', **tick_kw)
    ax1.tick_params(axis='x', colors='#CBD5E1')
    ax1.yaxis.grid(True, color='#334155', linestyle='--', linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#334155')

    legend = ax1.legend(
        loc='lower right', framealpha=0.25, edgecolor='#475569',
        labelcolor='white', fontsize=9,
    )

    # ── Panel 2: overall macro-avg bar chart ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#1E293B')

    overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    overall_values  = [overall[m] for m in overall_metrics]
    colors_ov       = [METRIC_COLORS[3], METRIC_COLORS[0],
                       METRIC_COLORS[1], METRIC_COLORS[2]]

    bars2 = ax2.bar(overall_metrics, overall_values, color=colors_ov, alpha=0.88,
                    width=0.55, zorder=3)
    for bar, val in zip(bars2, overall_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f'{val:.4f}',
            ha='center', va='bottom',
            color='white', fontsize=9, fontweight='bold',
        )

    ax2.set_title('Overall Model Scores (Macro Avg)', **title_kw)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Score', **label_kw)
    ax2.tick_params(axis='both', **tick_kw)
    ax2.tick_params(axis='x', colors='#CBD5E1', labelsize=9)
    ax2.yaxis.grid(True, color='#334155', linestyle='--', linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')

    # ── Panel 3: per-class F1 horizontal bars (visual ranking) ────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#1E293B')

    class_labels = [CLASS_DISPLAY.get(c, c).replace('\n', ' ') for c in class_names]
    f1_values    = [per_class[c]['F1-Score'] for c in class_names]
    class_colors = [PALETTE.get(c, '#64748B') for c in class_names]

    y_pos  = np.arange(len(class_names))
    hbars  = ax3.barh(y_pos, f1_values, color=class_colors, alpha=0.88,
                      height=0.55, zorder=3)
    for bar, val in zip(hbars, f1_values):
        ax3.text(
            min(val + 0.015, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', ha='left',
            color='white', fontsize=9, fontweight='bold',
        )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(class_labels, color='#CBD5E1', fontsize=9)
    ax3.set_xlim(0, 1.15)
    ax3.set_xlabel('F1-Score', **label_kw)
    ax3.set_title('F1-Score Ranking by Class', **title_kw)
    ax3.tick_params(axis='x', **tick_kw)
    ax3.xaxis.grid(True, color='#334155', linestyle='--', linewidth=0.6, zorder=0)
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#334155')

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        'Model Performance Scoreboard  ·  MobileNetV2 ALS Detector',
        color='white', fontsize=14, fontweight='bold', y=0.96,
    )

    # ── Accuracy watermark in top-right ───────────────────────────────────────
    fig.text(
        0.97, 0.96,
        f"Overall Accuracy  {overall['Accuracy']:.2%}",
        ha='right', va='top',
        color='#A78BFA', fontsize=11, fontweight='bold',
    )

    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n✅  Saved → {OUT_PATH}")
    return OUT_PATH


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model Performance Scoreboard")
    print("=" * 60)

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU  : {len(gpus)} device(s)")
    else:
        print("GPU  : none — using CPU")

    print(f"\nModel : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Params: {model.count_params():,}")

    print("\nLoading test split …")
    test_ds, class_names = load_test_split()

    print("\nRunning inference …")
    y_true, y_pred = evaluate(model, test_ds, class_names)

    per_class, overall = compute_metrics(y_true, y_pred, class_names)

    # ── Console scoreboard ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Overall Accuracy : {overall['Accuracy']:.4f}  ({overall['Accuracy']:.2%})")
    print(f"  Macro Precision  : {overall['Precision']:.4f}")
    print(f"  Macro Recall     : {overall['Recall']:.4f}")
    print(f"  Macro F1-Score   : {overall['F1-Score']:.4f}")
    print("=" * 60)
    print(f"\n  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "-" * 54)
    for name in class_names:
        m = per_class[name]
        print(f"  {name:<22} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1-Score']:>10.4f}")
    print()

    plot_scoreboard(per_class, overall, class_names)


if __name__ == '__main__':
    main()
