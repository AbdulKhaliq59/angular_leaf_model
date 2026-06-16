#!/usr/bin/env python3
"""
Generates logs/augmentation_counts.png — light-theme single diagram showing
dataset counts at every stage: raw → split → balanced → augmented views.

Run from angular_leaf_model/:
    python plot_augmentation_counts.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

os.makedirs('logs', exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
CLASSES      = ['Angular Leaf Spot', 'Healthy', 'Other Disease', 'Other Leaves']
C_COLORS     = ['#EF4444', '#22C55E', '#F97316', '#6366F1']

RAW          = [1107, 1115, 1103, 1103]
TRAIN_BAL    = [775,  780,  772,  772]
VAL_BAL      = [165,  165,  165,  165]
TEST         = [166,  168,  166,  166]

TOTAL_RAW    = 4428
TOTAL_TRAIN  = 3099
TOTAL_VAL    = 660
TOTAL_TEST   = 666
EPOCHS_HEAD  = 20
EPOCHS_FINE  = 40
TOTAL_EPOCHS = 60
TOTAL_VIEWS  = 185_940

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = '#F8FAFC'
PANEL       = '#FFFFFF'
BORDER      = '#E2E8F0'
TEXT_DARK   = '#1E293B'
TEXT_MID    = '#475569'
TEXT_LIGHT  = '#94A3B8'

COL_RAW     = '#334155'
COL_TRAIN   = '#16A34A'
COL_VAL     = '#2563EB'
COL_TEST    = '#DC2626'
COL_AUG     = '#7C3AED'

# ── Figure setup ──────────────────────────────────────────────────────────────
FIG_W, FIG_H = 20, 13
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor(BG)

# ── Drawing helpers ───────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc=PANEL, ec=BORDER, lw=1.2, pad=0.12, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
        clip_on=False,
    ))

def txt(x, y, s, **kw):
    ax.text(x, y, s, zorder=10, **kw)

def header_box(x, y, w, h, color, title, subtitle=''):
    rbox(x, y, w, h, fc=color, ec=color, lw=0, pad=0.10, zorder=3)
    ty = y + h * (0.65 if subtitle else 0.50)
    txt(x + w/2, ty, title, ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
    if subtitle:
        txt(x + w/2, y + h * 0.22, subtitle,
            ha='center', va='center', fontsize=8, color='white', alpha=0.88)

def class_row(x, y, w, h, cls, count, color, show_pct=None):
    rbox(x, y, w, h, fc=color + '18', ec=color + '55', lw=0.9, pad=0.08, zorder=2)
    ax.add_patch(plt.Circle((x + 0.22, y + h/2), 0.09, color=color, zorder=4))
    txt(x + 0.40, y + h/2, cls, ha='left', va='center',
        fontsize=8.5, color=TEXT_DARK)
    label = f"{count:,}" + (f"  ({show_pct}%)" if show_pct else "")
    txt(x + w - 0.18, y + h/2, label, ha='right', va='center',
        fontsize=9, fontweight='bold', color=color)

def total_row(x, y, w, h, total, color):
    rbox(x, y, w, h, fc=color + '22', ec=color, lw=1.5, pad=0.08, zorder=2)
    txt(x + 0.30, y + h/2, "TOTAL", ha='left', va='center',
        fontsize=9, fontweight='bold', color=color)
    txt(x + w - 0.18, y + h/2, f"{total:,}", ha='right', va='center',
        fontsize=12, fontweight='bold', color=color)

def arrow_h(x0, x1, y, label='', label_dy=0.22):
    ax.annotate('', xy=(x1 - 0.05, y), xytext=(x0 + 0.05, y),
                arrowprops=dict(arrowstyle='->', color=TEXT_LIGHT,
                                lw=1.4, mutation_scale=13), zorder=5)
    if label:
        txt((x0 + x1)/2, y + label_dy, label,
            ha='center', va='bottom', fontsize=8, color=TEXT_LIGHT)

def big_kpi(x, y, w, h, number, label, color):
    rbox(x, y, w, h, fc=color + '16', ec=color, lw=2.0, pad=0.14, zorder=2)
    txt(x + w/2, y + h * 0.60, f"{number:,}",
        ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    txt(x + w/2, y + h * 0.22, label,
        ha='center', va='center', fontsize=8, color=TEXT_MID)

def formula_row(x, y, w, h, label, value, color):
    rbox(x, y, w, h, fc=color + '14', ec=color + '55', lw=0.8, pad=0.08, zorder=2)
    txt(x + 0.22, y + h/2, label, ha='left', va='center',
        fontsize=8.5, color=TEXT_DARK)
    txt(x + w - 0.18, y + h/2, value, ha='right', va='center',
        fontsize=9.5, fontweight='bold', color=color)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(FIG_W/2, FIG_H - 0.38,
    "Dataset Size & Augmentation Count  —  Angular Leaf Spot Detection",
    ha='center', va='center', fontsize=18, fontweight='bold', color=TEXT_DARK)
txt(FIG_W/2, FIG_H - 0.78,
    "MobileNetV2 Transfer Learning  ·  4 Classes  ·  70 / 15 / 15 Split  ·  Online Augmentation",
    ha='center', va='center', fontsize=10, color=TEXT_MID)
ax.plot([0.5, FIG_W - 0.5], [FIG_H - 1.02, FIG_H - 1.02], color=BORDER, lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP  (just below title)
# ══════════════════════════════════════════════════════════════════════════════
KPI_Y   = FIG_H - 2.30
KPI_H   = 0.88
KPI_W   = 3.20
KPI_GAP = 0.28
KPI_X0  = (FIG_W - 5*KPI_W - 4*KPI_GAP) / 2

kpis = [
    (TOTAL_RAW,    "Total Raw Images",   COL_RAW),
    (TOTAL_TRAIN,  "Train (balanced)",   COL_TRAIN),
    (TOTAL_VAL,    "Val (balanced)",     COL_VAL),
    (TOTAL_TEST,   "Test (locked)",      COL_TEST),
    (TOTAL_VIEWS,  "Augmented Views\n(all epochs)", COL_AUG),
]
for i, (n, lbl, col) in enumerate(kpis):
    big_kpi(KPI_X0 + i*(KPI_W + KPI_GAP), KPI_Y, KPI_W, KPI_H, n, lbl, col)

ax.plot([0.5, FIG_W - 0.5], [KPI_Y - 0.22, KPI_Y - 0.22], color=BORDER, lw=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT  — 4 split columns + 1 augmentation column
# ══════════════════════════════════════════════════════════════════════════════
COL_TOP   = KPI_Y - 0.52   # top of content columns
HDR_H     = 0.54
ROW_H     = 0.50
ROW_GAP   = 0.06
TOT_H     = 0.52
N_ROWS    = 4

# Total height a column of 4 class rows + 1 total row + header occupies
COL_CONTENT_H = HDR_H + 0.12 + N_ROWS*(ROW_H + ROW_GAP) + 0.10 + TOT_H
COL_BOT   = COL_TOP - COL_CONTENT_H

COLS_W    = [3.0, 3.0, 3.0, 3.0]      # raw, train, val, test
AUG_W     = 4.2
ARROW_W   = 0.70

TOTAL_USED = sum(COLS_W) + 3*ARROW_W + AUG_W
MARGIN     = (FIG_W - TOTAL_USED) / 2

col_xs = []
x = MARGIN
for w in COLS_W:
    col_xs.append(x)
    x += w + ARROW_W
AUG_X = x

# ── helper to draw one full split column ─────────────────────────────────────
def split_column(x, w, color, title, subtitle, counts, total, note=''):
    header_box(x, COL_TOP - HDR_H, w, HDR_H, color, title, subtitle)
    ry = COL_TOP - HDR_H - 0.12
    for cls, cnt, col in zip(CLASSES, counts, C_COLORS):
        ry -= ROW_H
        class_row(x, ry, w, ROW_H, cls, cnt, col)
        ry -= ROW_GAP
    ry -= 0.10
    total_row(x, ry - TOT_H, w, TOT_H, total, color)
    if note:
        txt(x + w/2, ry - TOT_H - 0.28, note,
            ha='center', va='top', fontsize=7.5, color=TEXT_LIGHT)

split_column(col_xs[0], COLS_W[0], COL_RAW,
             "RAW DATASET", "all images collected",
             RAW, TOTAL_RAW)

split_column(col_xs[1], COLS_W[1], COL_TRAIN,
             "TRAIN  70 %", "balanced · cap = 605/class",
             TRAIN_BAL, TOTAL_TRAIN,
             "★ ALS weighted 1.2× in loss")

split_column(col_xs[2], COLS_W[2], COL_VAL,
             "VAL  15 %", "balanced · 90 / class",
             VAL_BAL, TOTAL_VAL,
             "no augmentation")

split_column(col_xs[3], COLS_W[3], COL_TEST,
             "TEST  15 %", "locked · never used in tuning",
             TEST, TOTAL_TEST,
             "no augmentation")

# ── arrows between columns ───────────────────────────────────────────────────
ARROW_Y = COL_TOP - HDR_H - N_ROWS*(ROW_H+ROW_GAP)/2 - 0.2
for i in range(len(COLS_W)):
    x0 = col_xs[i] + COLS_W[i]
    x1 = col_xs[i] + COLS_W[i] + ARROW_W if i < len(col_xs)-1 else AUG_X
    lbl = "70/15/15\nsplit" if i == 0 else ("10 augment\ntransforms" if i == 3 else "")
    arrow_h(x0, x1, ARROW_Y, lbl, label_dy=0.16)

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION COLUMN  (rightmost)
# ══════════════════════════════════════════════════════════════════════════════
header_box(AUG_X, COL_TOP - HDR_H, AUG_W, HDR_H,
           COL_AUG, "ONLINE AUGMENTATION", "unique random transform per epoch")

FRM_Y  = COL_TOP - HDR_H - 0.14
FRM_H  = ROW_H
FRM_G  = ROW_GAP

formulas = [
    ("Train samples per epoch",     f"{TOTAL_TRAIN:,}",            COL_TRAIN),
    ("Stage-1 epochs  (head only)", f"× {EPOCHS_HEAD}",            COL_AUG),
    ("Stage-2 epochs  (fine-tune)", f"× {EPOCHS_FINE}",            COL_AUG),
    ("Total training epochs",       f"= {TOTAL_EPOCHS}",           COL_AUG),
]

for label, val, color in formulas:
    FRM_Y -= FRM_H
    formula_row(AUG_X, FRM_Y, AUG_W, FRM_H, label, val, color)
    FRM_Y -= FRM_G

# divider + grand total
DIV_Y = FRM_Y - 0.14
ax.plot([AUG_X + 0.2, AUG_X + AUG_W - 0.2], [DIV_Y, DIV_Y], color=BORDER, lw=1.2)

GT_H = 1.0
GT_Y = DIV_Y - GT_H - 0.10
rbox(AUG_X, GT_Y, AUG_W, GT_H, fc=COL_AUG + '18', ec=COL_AUG, lw=2.0, pad=0.14)
txt(AUG_X + AUG_W/2, GT_Y + GT_H*0.64, f"{TOTAL_VIEWS:,}",
    ha='center', va='center', fontsize=24, fontweight='bold', color=COL_AUG)
txt(AUG_X + AUG_W/2, GT_Y + GT_H*0.24,
    "unique augmented training views\nacross all 60 epochs",
    ha='center', va='center', fontsize=8, color=TEXT_MID, linespacing=1.4)

# note below grand total
txt(AUG_X + AUG_W/2, GT_Y - 0.22,
    "Online aug — no extra files on disk\nEach image sees a fresh transform every epoch",
    ha='center', va='top', fontsize=7.5, color=TEXT_LIGHT, linespacing=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION STEPS LIST  (below split columns)
# ══════════════════════════════════════════════════════════════════════════════
STEPS_TOP = COL_BOT - 0.38
STEPS_X   = MARGIN
STEPS_W   = sum(COLS_W) + 2*ARROW_W   # span first four columns

rbox(STEPS_X, STEPS_TOP - 1.32, STEPS_W, 1.32,
     fc='#F0FDF4', ec=COL_TRAIN + '60', lw=1.0, pad=0.12)

txt(STEPS_X + 0.25, STEPS_TOP - 0.18,
    "10 Augmentation Steps applied to TRAIN only",
    ha='left', va='center', fontsize=9, fontweight='bold', color=COL_TRAIN)

STEP_LABELS = [
    "① Random Flip  (H+V)",
    "② Random Rotation  ±20°",
    "③ Random Zoom  ±20%",
    "④ Random Translation  ±10%",
    "⑤ Random Brightness  ±10%",
    "⑥ Random Contrast  ±10%",
    "⑦ Random Crop  → 201×201",
    "⑧ Resize  → 224×224",
    "⑨ Gaussian Noise  σ=2.0",
    "⑩ MobileNetV2 Preprocess  [−1, 1]",
]

cols_per_row = 5
step_col_w = STEPS_W / cols_per_row
for i, s in enumerate(STEP_LABELS):
    col_i  = i % cols_per_row
    row_i  = i // cols_per_row
    sx = STEPS_X + col_i * step_col_w + 0.18
    sy = STEPS_TOP - 0.50 - row_i * 0.36
    txt(sx, sy, s, ha='left', va='center',
        fontsize=8, color=TEXT_MID)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([0.5, FIG_W - 0.5], [0.72, 0.72], color=BORDER, lw=0.8)
txt(FIG_W/2, 0.44,
    "Val & Test receive MobileNetV2 preprocessing only  ·  "
    "ALS class weighted 1.2× in training loss  ·  "
    "Input resolution: 224 × 224 px  ·  Batch size: 16",
    ha='center', va='center', fontsize=8, color=TEXT_LIGHT)

# ── Save ─────────────────────────────────────────────────────────────────────
OUT = 'logs/augmentation_counts.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✅  Saved → {OUT}")
