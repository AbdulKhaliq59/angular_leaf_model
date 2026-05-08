# Inference explainer — How the system decides what a photo contains

This document explains, with concrete examples, how a captured image is processed to decide:

1. Is the image a leaf (vs other things)?
2. If a leaf: is it a bean leaf (vs other plant leaves)?
3. If a bean leaf: is it Angular Leaf Spot (ALS), another disease, or healthy?

This follows the production pipeline implemented in `model_service.py` (Stages 1–4).

---

## Pipeline summary (4 stages)

- **Stage 1 — Image quality**: check minimum size and blur score (Laplacian variance). Reject images that are too small or too blurry.
- **Stage 2 — Leaf colour detection**: HSV colour masks detect plant-coloured pixels; require a minimum plant-pixel ratio to consider the image a leaf.
- **Stage 3 — Bean-leaf shape validation**: contour analysis tests solidity and elongation to filter non-bean leaves (e.g., grass, needles) and fragmented backgrounds.
- **Stage 4 — Disease classification**: the ML model (sigmoid for binary or softmax for multi-class) assigns disease/healthy labels and a confidence score; thresholds produce final labels and an uncertainty band.

Each stage either rejects the image or passes it to the next stage.

---

## Stage details with concrete examples

All example numeric thresholds below match the defaults in `model_service.py`.

### Stage 1: Image quality

- Checks:
  - Minimum side length: `MIN_IMAGE_PIXELS = 100` px
  - Blur: compute Laplacian variance; `BLUR_THRESHOLD = 60.0`

Example A — too small:

- Image dims: 80×120 → min side = 80 < 100 → result: `low_quality` (rejected)

Example B — blurry:

- Laplacian variance computed on grayscale = 25.4 < 60 → `low_quality` (rejected)

If Stage 1 rejects, final output fields resemble:

{
  "predicted_class": "low_quality",
  "status": "rejected",
  "confidence": 0.0,
  "interpretation": "image too blurry (score=25.4)"
}

### Stage 2: Leaf colour detection

- Method: convert to HSV, compute two masks (green and yellow-green) then union them. Compute `plant_ratio = plant_pixels / total_pixels`. Require `plant_ratio >= LEAF_COLOR_RATIO_MIN` (default 0.08).

Example C — not a leaf (random object / pavement):

- total_pixels = 224×224 = 50176
- plant_pixels = 320 → plant_ratio = 0.006 < 0.08 → result: `not_leaf`

Output example:

{
  "predicted_class": "not_leaf",
  "status": "not_leaf",
  "is_leaf": false,
  "plant_ratio": 0.006,
  "interpretation": "insufficient plant-coloured pixels"
}

Example D — green object (leaf-like color) passes Stage 2:

- plant_pixels = 7200 → plant_ratio = 0.144 ≥ 0.08 → pass to Stage 3

### Stage 3: Bean-leaf shape validation

- Method: create a broad plant mask (wider HSV range), perform morphological cleanup, find contours, take the largest contour and compute:
  - Area fraction (contour area relative to image)
  - Solidity = area / convex_hull_area (require ≥ `BEAN_LEAF_MIN_SOLIDITY`, default 0.35)
  - Elongation = max(width,height)/min(width,height) (require ≤ `BEAN_LEAF_MAX_ELONGATION`, default 7.0)

Example E — not bean-like (grass clump):

- largest contour area = 500 px, image area = 50176 → area fraction = 0.01% < 3% → `not_bean_leaf`

Example F — bean-like leaf:

- largest contour area = 9000 px (≥ 3% of image)
- hull_area = 11000 → solidity = 9000/11000 = 0.818 ≥ 0.35
- bounding box w,h = 140,110 → elongation = 140/110 = 1.27 ≤ 7.0 → pass to Stage 4

If Stage 3 rejects, output:

{
  "predicted_class": "not_bean_leaf",
  "status": "not_leaf",
  "is_leaf": true,
  "interpretation": "leaf-like color but shape not bean-like (too small / low solidity)"
}

### Stage 4: Disease classification (ML model)

- Model types:
  - Binary model (2-class): single sigmoid output s ∈ (0,1) where high values ≈ `angular_leaf_spot`, low values ≈ `healthy`.
  - Multi-class model (3+ classes): softmax vector p over classes (e.g., `angular_leaf_spot`, `bean_rust`, `healthy`, `other_leaves`).

- Binary thresholds used in service (example defaults):
  - `HEALTHY_THRESHOLD = 0.88` ⇒ if s ≤ 0.88 → `healthy`
  - `ALS_THRESHOLD = 0.96` ⇒ if s ≥ 0.96 → `angular_leaf_spot`
  - `0.88 < s < 0.96` ⇒ `other_disease` (uncertain band)

Example G — binary model confident ALS:

- sigmoid s = 0.981 ≥ 0.96 ⇒ predicted `angular_leaf_spot`

Output example:

{
  "predicted_class": "angular_leaf_spot",
  "status": "unhealthy",
  "is_leaf": true,
  "confidence": 0.981,
  "result": "Angular leaf spot detected",
  "threshold": 0.96,
  "interpretation": "high-confidence sigmoid above ALS_THRESHOLD"
}

Example H — binary model ambiguous:

- sigmoid s = 0.92 → lies in uncertainty band → predicted `other_disease` (flag for manual review)

Example I — multi-class model:

- softmax output p = [0.03, 0.02, 0.88, 0.07] with class mapping [ALS, bean_rust, healthy, other_leaves] → argmax = index 2 → `healthy` with confidence 0.88

If multi-class predicts `other_leaves` with high confidence, that overrides Shape check (meaning the model learned visual patterns that match non-bean leaves even though color/shape partially matched).

---

## End-to-end example (concise)

Image captured → decode to BGR numpy.

1. Stage 1: dims 224×224 OK, Laplacian var = 120 → pass.
2. Stage 2: plant_ratio = 0.14 ≥ 0.08 → pass.
3. Stage 3: largest contour area = 9000 px, solidity = 0.82, elongation = 1.3 → pass.
4. Stage 4: model returns sigmoid s = 0.981 → final label `angular_leaf_spot` (confidence 0.981). The service returns JSON summarising all stages and the final interpretation.

Sample final payload returned by `AngularLeafSpotModel.predict(image)`:

{
  "predicted_class": "angular_leaf_spot",
  "status": "unhealthy",
  "health_status": "UNHEALTHY",
  "is_leaf": true,
  "confidence": 0.981,
  "result": "Angular leaf spot detected",
  "threshold": 0.96,
  "interpretation": "passed quality/leaf/shape checks; sigmoid=0.981 >= ALS_THRESHOLD"
}

---

## Notes & practical advice for deployment and research

- The pipeline uses conservative thresholds early (shape + colour) to reject non-leaf images cheaply before invoking the model.
- The uncertainty band (e.g., 0.88–0.96) is intentionally used to defer ambiguous cases to human review and to reduce harmful false positives.
- If you need fewer false negatives (catch more disease), lower `ALS_THRESHOLD` or tune thresholds by optimizing recall on a held-out test set.
- For rigorous reporting in a thesis: provide the dataset used for threshold selection, the ROC curve (binary) or per-class confusion matrix (multi-class), and the percentage of validation samples falling into the uncertainty band.

---

If you want, I can:

- Add a script `compute_thresholds.py` that runs the model on a validation set, computes ROC/AUC and candidate thresholds, and saves plots.
- Add an example Curl/HTTP request to the backend showing inference JSON responses.

File created: `INFERENCE_EXPLAINER.md` in the `angular_leaf_model` folder.
