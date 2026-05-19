# Model Reliability Plan — Angular Leaf Spot Detection

> **Purpose:** Document every change needed to make the model reliable and predictable before any code is touched.
> No implementation is included here — this is a full audit and improvement roadmap.

---

## 1. Current State Summary

| Item | Current Value | Status |
|---|---|---|
| Architecture | MobileNetV2 + custom head | OK |
| Total images | 1,428 across 4 classes | **Too small** |
| Train/val split | 80/20 (no test set) | **Missing test set** |
| Augmentation ops | 5 (flip, rotate, zoom, brightness, contrast) | **Incomplete** |
| Preprocessing (train) | `mobilenet_v2.preprocess_input` → [-1, 1] | OK |
| Preprocessing (inference) | `/255.0` → [0, 1] | **BUG — mismatch** |
| Metrics tracked | accuracy only (multi-class mode) | **Insufficient** |
| Threshold calibration | Hardcoded (not calibrated on test data) | **Unreliable** |
| Dataset shuffle/cache order | cache → shuffle (augmentation frozen) | **Bug** |
| Stage 2 HSV range | Excludes brown/red disease lesions | **False rejections** |

---

## 2. Critical Bug — Preprocessing Mismatch

**This is the single most damaging issue.**

`train_model.py` applies `mobilenet_v2.preprocess_input` which converts pixels to `[-1, 1]`.  
`model_service.py → preprocess_image()` divides by `255.0` which produces `[0, 1]`.

The model was trained expecting `[-1, 1]` but receives `[0, 1]` at inference time.
This shifts every pixel value by ~0.5 in absolute terms, causing the model to output
unreliable probabilities regardless of image content — this alone explains most
unpredictability seen in production.

**Fix required in `model_service.py`:**  
Replace `resized / 255.0` with `tf.keras.applications.mobilenet_v2.preprocess_input(resized)`
(do not divide by 255 at all — `preprocess_input` handles the full conversion).

---

## 3. Dataset — Size and Quality

### 3.1 Current class sizes

| Class | Current | Minimum recommended | Gap |
|---|---|---|---|
| angular_leaf_spot | 347 | 1,000 | −653 |
| healthy | 343 | 1,000 | −657 |
| bean_rust | 436 | 1,000 | −564 |
| other_leaves | 302 | 1,000 | −698 |
| **Total** | **1,428** | **4,000** | **−2,572** |

A production agricultural classifier targeting ≥90 % F1 per class typically needs
**1,000–2,000 images per class**. With ~340 per class the model is forced to rely
heavily on augmentation and transfer learning, which introduces variance.

### 3.2 Where to source additional images

| Class | Recommended sources |
|---|---|
| angular_leaf_spot | PlantVillage dataset (Kaggle), IITA bean disease image archives, field photos from local agricultural extension offices |
| healthy | PlantVillage healthy bean split, Roboflow bean datasets |
| bean_rust | PlantVillage bean rust split (already used via Hugging Face — increase count from 436 to 1,000+) |
| other_leaves | iNaturalist (already used — increase from 302 to 1,000+; add more species: cassava, sorghum, wheat, soybean) |

### 3.3 Data quality requirements

- Minimum image resolution: **300×300 px** (currently accepting 100×100 — too low for reliable feature extraction)
- Remove duplicate/near-duplicate images — run perceptual hash dedup before training
- Ensure ALS images cover **early, mid, and late stage** lesions (currently unknown distribution)
- Ensure healthy images include **young leaves, mature leaves, and slightly stressed leaves** (drought, nutrient deficiency) so the model learns that not every imperfect leaf is diseased
- Include images taken under **different lighting** (direct sun, shade, overcast) and **different backgrounds** (soil, other plants, hand-held)

### 3.4 Labeling consistency

Add a labeling guideline document that defines:
- What "early ALS" looks like vs. "healthy with minor spots"
- The minimum lesion coverage % to label as ALS vs. healthy
- Whether partially diseased leaves count as healthy or diseased

Without this, label noise from ambiguous cases directly reduces model precision.

---

## 4. Augmentation Pipeline — Gaps to Fill

### 4.1 Current pipeline

```
RandomFlip(horizontal_and_vertical)
RandomRotation(0.3)      ← ±30 % of 2π  ≈ ±108°
RandomZoom(0.2)          ← ±20 %
RandomBrightness(0.2)    ← ±20 % brightness
RandomContrast(0.2)      ← ±20 % contrast
```

### 4.2 Missing augmentations and why each matters

| Augmentation | Config to add | Why it matters |
|---|---|---|
| `RandomTranslation` | height_factor=0.1, width_factor=0.1 | Field photos have random framing; the leaf is rarely perfectly centered |
| `RandomHue` | factor=0.05 | Different phone cameras produce slightly different color temperatures; helps generalize across devices |
| `RandomSaturation` | (use `tf.image.adjust_saturation` in a Lambda) lower=0.7, upper=1.3 | Saturation varies with ambient light and camera auto-processing |
| Gaussian noise | std=0.02–0.05 applied to normalized tensor | Simulates camera sensor noise, especially from lower-end phones common in target user demographic |
| `RandomCrop` | crop to 90–95 % of original then resize back to 224 | Forces the model to learn local lesion features rather than relying on the whole leaf boundary |
| `Cutout / CoarseDropout` | 1–3 rectangles, each up to 15 % of image area | Simulates partial occlusion (fingers, other leaves); improves robustness to partially visible lesions |
| Horizontal + vertical shear | ±10° | Simulates perspective distortion from scanning at an angle; very common when holding a phone over a leaf |

### 4.3 Augmentation strength calibration

Each added augmentation should be validated against a held-out test set (see §6)
to confirm it improves generalization. Adding too many at high strength can
over-augment and reduce accuracy on realistic inputs.

Recommended sweep:
1. Add `RandomTranslation` → measure Δval_accuracy
2. Add Gaussian noise → measure Δval_accuracy
3. Add `Cutout` → measure Δval_accuracy
4. Only keep augmentations that produce zero or positive delta on the test set

### 4.4 Fix — Cache/shuffle order

**Current (wrong):**
```
train_raw.map(augment).cache().shuffle().prefetch()
```
Because `.cache()` comes before `.shuffle()`, augmentations are computed once and
frozen — every epoch sees the same augmented versions in a different order.
This defeats the purpose of augmentation.

**Correct order:**
```
train_raw.cache().shuffle(buffer_size).map(augment).prefetch()
```
Cache the raw images (fast disk reads), shuffle them, then apply augmentation fresh
every time an image is drawn. This ensures each epoch generates different augmented
variants of every image.

---

## 5. Model Architecture — Improvements

### 5.1 Fine-tuning depth

Currently only the last 30 layers of MobileNetV2 (~154 total) are unfrozen in Stage 2.
This is ~20% of the network. Because leaf disease is a domain-specific task (not
ImageNet-like), more layers should be adapted.

**Recommendation:** Unfreeze from layer `-60` (top ~40%) rather than `-30`.
Use the same low learning rate (1e-5) with a warm-up to avoid destroying early features.

### 5.2 Head architecture options

Current head: `GAP → BN → Dense(256) → Dropout(0.4) → Dense(64) → Dropout(0.2) → output`

Consider adding `L2 regularization (kernel_regularizer=l2(1e-4))` to both Dense layers.
This penalizes large weights and reduces overfitting on the small dataset.

### 5.3 Alternative backbone — EfficientNetB0

MobileNetV2 was designed for mobile inference (speed-over-accuracy trade-off).
EfficientNetB0 has a similar parameter count but typically achieves ~2-3% higher
top-1 accuracy on fine-grained classification tasks.

If retraining is possible, benchmarking both architectures on the full dataset
is worth the effort. EfficientNetB0 uses its own preprocessing (`efficientnet.preprocess_input`)
which must be consistently applied in both training and inference.

### 5.4 Binary vs. multi-class decision

The current binary model (sigmoid threshold) is fundamentally limited:
- It can only distinguish ALS vs. healthy
- Bean rust produces a confidence score with no clear semantic meaning
- The "uncertain band" (0.88–0.96) has no training signal

**Recommendation:** Move entirely to the 4-class softmax model and retire the binary path.
The `bean_rust` and `other_leaves` classes give the model explicit training signal
for every real-world case, eliminating the hard-coded uncertainty band.

---

## 6. Evaluation — What is Currently Missing

### 6.1 No dedicated test set

The current pipeline uses 80% train / 20% val. That validation set is used for:
- `EarlyStopping` decisions
- `ModelCheckpoint` best-model selection
- Final `evaluate()` reporting

All three uses contaminate the validation accuracy — it is an optimistic estimate.

**Fix:** Reserve an additional 10% as a **locked test set** before training starts.
Never use it during training. Use it only once at the end to report final metrics.

Suggested split: **70% train / 15% val / 15% test**

### 6.2 Metrics to add beyond accuracy

| Metric | Why |
|---|---|
| Per-class Precision | What fraction of "ALS" predictions are actually ALS — important for alert fatigue |
| Per-class Recall (Sensitivity) | What fraction of actual ALS cases are caught — critical for a disease-detection tool; a missed ALS case is far worse than a false alarm |
| Per-class F1 | Harmonic mean; the single number to optimize for imbalanced classes |
| Macro-average F1 | Mean F1 across all classes, treats each class equally |
| Confusion matrix | Shows exactly which classes are being confused with each other |
| AUC-ROC (one-vs-rest) | Area under curve; measures separation quality independent of threshold |

Generate a full `sklearn.metrics.classification_report` and plot the confusion matrix
after every training run.

### 6.3 Per-stage rejection rate tracking

Stage 1–3 currently reject silently. In production it is not known what % of real
field images are being rejected at each stage without ever reaching the ML model.
If Stage 3 (bean leaf shape) rejects 30% of legitimate bean leaf photos, the whole
pipeline is unreliable for users.

Add a logging hook that records: total predictions / stage1_rejects / stage2_rejects /
stage3_rejects / ml_predictions per day. Review it regularly.

---

## 7. Inference Threshold Calibration

### 7.1 Binary model thresholds are not calibrated

`ALS_THRESHOLD = 0.96` and `HEALTHY_THRESHOLD = 0.88` were likely chosen manually.
The gap between them (0.88–0.96) represents a large zone where the model returns
`other_disease` — a class it has never been trained on.

On a proper test set, plot the sigmoid distribution for ALS and healthy images.
Set `ALS_THRESHOLD` at the point that achieves ~95% recall for ALS (disease not missed).
Set `HEALTHY_THRESHOLD` so that false-positive rate on healthy leaves is acceptable.

### 7.2 Multi-class minimum confidence

`MIN_CONFIDENCE = 0.50` means anything below 50% softmax probability is rejected
as `other_disease`. With 4 classes, uniform probability is 25%, so 50% is a
reasonable floor. However it should be validated: if the model assigns 49% to ALS
and 47% to healthy, calling it `other_disease` is misleading.

Consider reporting the top-2 classes and probabilities rather than a single rejection
when the max confidence is between 0.40–0.60.

---

## 8. Stage 2 & 3 — HSV Pre-filter Issues

### 8.1 Stage 2 misses heavily diseased leaves

The current green detection range is H 25–90 (green) and H 15–25 (yellow-green).

Angular leaf spot creates brown-red angular lesions. A leaf with large ALS coverage
will have significant brown/orange pixels (H 0–20) that fall **outside** both ranges.
A heavily diseased leaf could fail Stage 2 and be returned as `not_leaf` — the worst
possible false negative for a disease detection tool.

**Fix needed:** Widen the `yellow_green_mask` down to H 10 (to catch early yellow lesions)
and add a separate `brown_mask` for H 0–15 with S > 50 (brownish disease lesions on leaves).
If the combined green + yellow + brown pixels exceed the threshold, pass Stage 2.

### 8.2 Stage 3 solidity threshold is too permissive

`BEAN_LEAF_MIN_SOLIDITY = 0.35` allows shapes where only 35% of the convex hull area
is filled — this is extremely irregular. A well-formed bean leaf photograph typically
has solidity ≥ 0.60.

However, because ALS lesions can fragment the leaf edge and reduce the measured solidity,
this threshold was likely lowered to avoid false rejections. The right fix is to use
the unfragmented bean leaf images to establish a proper lower bound empirically,
not to use a permissive hardcoded value.

**Fix needed:** Measure solidity on 50+ confirmed bean leaf images at different disease
stages and set the threshold at the 5th percentile of that distribution.

---

## 9. Training Stability

### 9.1 Batch size

Current `BATCH_SIZE = 16` is appropriate for the dataset size but could be increased
to 32 if GPU memory allows. Larger batches stabilize the gradient estimates during
fine-tuning (Stage 2) where the learning rate is already very low (1e-5).

### 9.2 Warmup for fine-tuning

Stage 2 immediately starts at `lr=1e-5`. Adding a linear warmup over the first 2–3
epochs (e.g., from 1e-7 to 1e-5) prevents large gradient updates from destabilizing
the freshly unfrozen layers.

### 9.3 Label smoothing

Replace `sparse_categorical_crossentropy` with `sparse_categorical_crossentropy`
with `label_smoothing=0.1`. This prevents the model from becoming overconfident
(assigning probabilities close to 1.0) on training images — directly improving
calibration and reducing the severity of incorrect high-confidence predictions.

### 9.4 k-fold cross-validation

With only 1,428 images, a single 70/15/15 split can produce misleading results
depending on which images land in each fold.

Run **5-fold cross-validation** on the full dataset and report mean ± std F1 per class.
This is the only way to know if reported accuracy is real or lucky.

---

## 10. Summary — Priority Order

| Priority | Change | Impact |
|---|---|---|
| **P0 — Fix now** | Preprocessing mismatch (`/255` vs `preprocess_input`) in `model_service.py` | Fixes root cause of unpredictable inference |
| **P0 — Fix now** | Cache/shuffle order in `load_datasets()` | Enables real augmentation diversity per epoch |
| **P1 — Before next retrain** | Expand dataset to 1,000 images/class | Largest accuracy improvement possible |
| **P1 — Before next retrain** | Add dedicated 15% test set, track per-class F1 + confusion matrix | Makes accuracy claims trustworthy |
| **P1 — Before next retrain** | Add RandomTranslation, Gaussian noise, RandomCrop, Cutout | Better generalization to real field photos |
| **P1 — Before next retrain** | Move to 4-class softmax only, retire binary path | Removes the uncalibrated uncertainty band |
| **P2 — Next iteration** | Widen Stage 2 HSV mask to include brown disease lesions | Prevents false `not_leaf` rejections on heavily diseased leaves |
| **P2 — Next iteration** | Calibrate thresholds on test set instead of hardcoding | Reliable confidence scores |
| **P2 — Next iteration** | Add L2 regularization to Dense layers | Reduces overfitting on small dataset |
| **P3 — Future** | Benchmark EfficientNetB0 vs MobileNetV2 | Potential 2–3% accuracy gain |
| **P3 — Future** | k-fold cross-validation | Reliable accuracy estimates |
| **P3 — Future** | Per-stage rejection rate logging in production | Catches silent pipeline failures |
