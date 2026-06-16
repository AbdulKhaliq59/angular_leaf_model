#!/usr/bin/env python3
"""
Generates logs/augmentation_pipeline.png — a single light-theme diagram
showing every augmentation step applied to each dataset split.

Run from angular_leaf_model/:
    python plot_augmentation_pipeline.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs('logs', exist_ok=True)

IMG_SIZE = 224

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = '#F8FAFC'
PANEL_BG    = '#FFFFFF'
BORDER      = '#E2E8F0'
TEXT_DARK   = '#1E293B'
TEXT_MID    = '#475569'
TEXT_LIGHT  = '#94A3B8'

COL_SPATIAL  = '#0EA5E9'   # sky-blue  — spatial transforms
COL_COLOR    = '#F59E0B'   # amber     — colour / intensity
COL_NOISE    = '#8B5CF6'   # violet    — noise
COL_PREPROC  = '#10B981'   # emerald   — preprocessing
COL_HEADER_TRAIN = '#16A34A'
COL_HEADER_VAL   = '#2563EB'
COL_HEADER_TEST  = '#DC2626'

# ── Step definitions ──────────────────────────────────────────────────────────
TRAIN_STEPS = [
    (COL_SPATIAL,  "Random Flip",         "horizontal + vertical"),
    (COL_SPATIAL,  "Random Rotation",     "factor = 0.2  (±20°)"),
    (COL_SPATIAL,  "Random Zoom",         "factor = 0.2  (±20 %)"),
    (COL_SPATIAL,  "Random Translation",  "height ±10 %,  width ±10 %"),
    (COL_COLOR,    "Random Brightness",   "factor = 0.1  (±10 %)"),
    (COL_COLOR,    "Random Contrast",     "factor = 0.1  (±10 %)"),
    (COL_SPATIAL,  "Random Crop",         f"→ {int(IMG_SIZE*0.9)}×{int(IMG_SIZE*0.9)} px  (90 %)"),
    (COL_SPATIAL,  "Resize",              f"→ {IMG_SIZE}×{IMG_SIZE} px"),
    (COL_NOISE,    "Gaussian Noise",      "mean = 0.0,  σ = 2.0"),
    (COL_PREPROC,  "MobileNetV2 Preprocess", "pixels → [−1, 1]"),
]

EVAL_STEPS = [
    (COL_PREPROC,  "MobileNetV2 Preprocess", "pixels → [−1, 1]"),
]

LEGEND_ITEMS = [
    (COL_SPATIAL,  "Spatial transform"),
    (COL_COLOR,    "Colour / intensity"),
    (COL_NOISE,    "Noise injection"),
    (COL_PREPROC,  "Preprocessing"),
]

# ── Layout constants ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 11
BOX_H    = 0.52
BOX_GAP  = 0.14
COL_W    = 4.2
PADDING  = 0.22

COLS = [
    dict(label="TRAIN",  steps=TRAIN_STEPS, hdr=COL_HEADER_TRAIN,
         note="Augmented every epoch  •  10 transforms"),
    dict(label="VAL",    steps=EVAL_STEPS,  hdr=COL_HEADER_VAL,
         note="No augmentation  •  preprocess only"),
    dict(label="TEST",   steps=EVAL_STEPS,  hdr=COL_HEADER_TEST,
         note="No augmentation  •  preprocess only\n(locked — never used for tuning)"),
]

# x-centres for the three columns
X_CENTRES = [2.2, 7.5, 12.8]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor(BG)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.42, "Data Augmentation Pipeline",
        ha='center', va='center', fontsize=19, fontweight='bold', color=TEXT_DARK)
ax.text(FIG_W / 2, FIG_H - 0.88,
        "Angular Leaf Spot Detection  —  MobileNetV2 Transfer Learning",
        ha='center', va='center', fontsize=10.5, color=TEXT_MID)

# thin separator under title
ax.plot([0.6, FIG_W - 0.6], [FIG_H - 1.12, FIG_H - 1.12],
        color=BORDER, linewidth=1.2)

TOP_Y = FIG_H - 1.55   # top of column content area


def draw_step_box(cx, y, color, title, detail, idx):
    """Draw a single augmentation step card."""
    x0 = cx - COL_W / 2 + PADDING
    w  = COL_W - 2 * PADDING

    # shadow
    shadow = FancyBboxPatch(
        (x0 + 0.04, y - BOX_H - 0.04), w, BOX_H,
        boxstyle="round,pad=0.06", linewidth=0,
        facecolor='#CBD5E1', zorder=1,
    )
    ax.add_patch(shadow)

    # main box
    box = FancyBboxPatch(
        (x0, y - BOX_H), w, BOX_H,
        boxstyle="round,pad=0.06", linewidth=1.2,
        edgecolor=color, facecolor=PANEL_BG, zorder=2,
    )
    ax.add_patch(box)

    # left accent bar
    bar = FancyBboxPatch(
        (x0, y - BOX_H), 0.12, BOX_H,
        boxstyle="round,pad=0.0", linewidth=0,
        facecolor=color, zorder=3,
    )
    ax.add_patch(bar)

    # step number circle
    circ = plt.Circle((x0 + 0.34, y - BOX_H / 2), 0.18,
                       color=color, zorder=4)
    ax.add_patch(circ)
    ax.text(x0 + 0.34, y - BOX_H / 2, str(idx),
            ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='white', zorder=5)

    # step name
    ax.text(x0 + 0.62, y - BOX_H * 0.35, title,
            ha='left', va='center', fontsize=9, fontweight='bold',
            color=TEXT_DARK, zorder=5)

    # parameters
    ax.text(x0 + 0.62, y - BOX_H * 0.72, detail,
            ha='left', va='center', fontsize=7.8,
            color=TEXT_MID, zorder=5)


def draw_arrow(cx, y_from, y_to):
    """Downward connector arrow between boxes."""
    ax.annotate('',
        xy=(cx, y_to + 0.01),
        xytext=(cx, y_from - 0.01),
        arrowprops=dict(
            arrowstyle='->', color=TEXT_LIGHT,
            lw=1.1, mutation_scale=10,
        ),
        zorder=1,
    )


# ── Draw each column ──────────────────────────────────────────────────────────
for col, cx in zip(COLS, X_CENTRES):
    # Column header pill
    hdr_x = cx - COL_W / 2 + PADDING
    hdr_w = COL_W - 2 * PADDING
    hdr_y = TOP_Y

    hdr_box = FancyBboxPatch(
        (hdr_x, hdr_y - 0.52), hdr_w, 0.52,
        boxstyle="round,pad=0.06", linewidth=0,
        facecolor=col['hdr'], zorder=2,
    )
    ax.add_patch(hdr_box)
    ax.text(cx, hdr_y - 0.26, col['label'],
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='white', zorder=3)

    # Note below header
    note_y = hdr_y - 0.68
    ax.text(cx, note_y, col['note'],
            ha='center', va='top', fontsize=7.5,
            color=TEXT_MID, linespacing=1.5, zorder=3)

    note_lines = col['note'].count('\n') + 1
    step_start_y = note_y - note_lines * 0.20 - 0.20

    # Steps
    for i, (color, title, detail) in enumerate(col['steps']):
        box_top = step_start_y - i * (BOX_H + BOX_GAP)
        draw_step_box(cx, box_top, color, title, detail, i + 1)

        if i < len(col['steps']) - 1:
            draw_arrow(cx, box_top - BOX_H, box_top - BOX_H - BOX_GAP)

# ── Legend ────────────────────────────────────────────────────────────────────
LEG_X = 0.65
LEG_Y = 1.12
ax.text(LEG_X, LEG_Y, "Legend", ha='left', va='bottom',
        fontsize=9, fontweight='bold', color=TEXT_DARK)

for k, (color, label) in enumerate(LEGEND_ITEMS):
    bx = LEG_X + k * 3.6
    patch = FancyBboxPatch((bx, LEG_Y - 0.38), 0.28, 0.28,
                            boxstyle="round,pad=0.04", linewidth=0,
                            facecolor=color, zorder=2)
    ax.add_patch(patch)
    ax.text(bx + 0.38, LEG_Y - 0.24, label,
            ha='left', va='center', fontsize=8.5, color=TEXT_MID)

# ── Footer ────────────────────────────────────────────────────────────────────
ax.plot([0.6, FIG_W - 0.6], [0.72, 0.72], color=BORDER, linewidth=0.8)
ax.text(FIG_W / 2, 0.44,
        "Augmentation applied only to TRAIN split  •  "
        "Val & Test receive preprocessing only  •  "
        f"Input resolution: {IMG_SIZE}×{IMG_SIZE} px",
        ha='center', va='center', fontsize=8, color=TEXT_LIGHT)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = 'logs/augmentation_pipeline.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✅  Saved → {OUT}")
