# Model training notes — Angular Leaf Spot Detection

This document describes how the model was trained, the dataset layout and preprocessing, the training strategy, and the mathematical formulas you can quote in a thesis.

**Repository:** angular_leaf_model

**Quick commands**

- Train: `python train_model.py`
- Trained model saved to: `models/beenleaf_model.h5`

**Contents**
- **Overview**
- **Dataset and splits**
- **Preprocessing & augmentation**
- **Model architecture & outputs**
- **Losses, metrics & formulas**
- **Training schedule**
- **Inference thresholds & mapping**

**Overview**

We use transfer learning with MobileNetV2 (ImageNet weights) and a small custom head. The pipeline supports both binary (2-class) and multi-class setups automatically depending on the number of subfolders in `data/`.

**Dataset and splits**

- Expected structure (examples):
  - `data/angular_leaf_spot/`
  - `data/healthy/`
  - `data/other_leaves/` (optional, recommended)
- The training script uses an 80/20 train/validation split via Keras `image_dataset_from_directory` with a fixed seed for reproducibility.
- Class names are read from the folder names and sorted by Keras directory order. For multi-class experiments, include an `other_leaves` folder to teach the model to reject non-bean leaves.

Class balance

We compute class weights from training sample counts to counter class imbalance. The implemented formula is:

$$
w_c = \frac{N}{C \cdot \max(n_c, 1)}
$$

where:
- $w_c$ is the weight for class $c$
- $N$ is the total number of training samples
- $C$ is the number of classes
- $n_c$ is the number of samples in class $c$ (clamped to at least 1)

This matches the code in `train_model.py` and is passed to `model.fit(..., class_weight=class_weight)`.

**Preprocessing & augmentation**

- Input size: 224 × 224 (MobileNetV2 standard)
- Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input` (scales pixels to MobileNetV2 expected range)
- Training augmentation (applied only to the training set): random flips, rotation, zoom, brightness and contrast variations, implemented as a Keras `Sequential` augmentation block.

These augmentations increase robustness to viewpoint, lighting, and small scale differences.

**Model architecture & outputs**

- Base: MobileNetV2 (include_top=False, pretrained on ImageNet)
- Head: GlobalAveragePooling2D → BatchNorm → Dense(256, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.2) → output layer
- Output type:
  - Binary (2-class): single neuron with sigmoid activation → output in $(0,1)$
  - Multi-class (3+ classes): $C$ neurons with softmax activation → output is a probability vector

Sigmoid function (binary):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Softmax (multi-class):

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

**Losses, metrics & formulas**

- Binary loss: binary cross-entropy

$$
\mathcal{L}_{binary} = -\left(y\log p + (1-y)\log(1-p)\right)
$$

where $y\in\{0,1\}$ and $p=\sigma(z)$ is the predicted probability for the positive class.

- Multi-class loss: sparse categorical cross-entropy (labels provided as integer indices):

$$
\mathcal{L}_{sparse\_cat} = -\log p_{y}
$$

where $p_{y}$ is the softmax probability assigned to the true class index $y$.

- Metrics: accuracy for all runs; AUC (ROC area) is added for binary experiments to better capture separability regardless of threshold.

**Training schedule**

Two-stage training is used:

1. Stage 1 — Train the new head only (base frozen)
   - Optimizer: Adam, learning rate = 1e-3
   - Epochs: controlled by `EPOCHS_HEAD` (default 15)
   - Callbacks: ModelCheckpoint (save best), EarlyStopping (monitor best metric), ReduceLROnPlateau

2. Stage 2 — Fine-tune top layers of base
   - Unfreeze the last ~30 layers of the MobileNetV2 base and recompile
   - Optimizer: Adam, learning rate = 1e-5
   - Epochs: controlled by `EPOCHS_FINE` (default 25)

This schedule trains the classification head first to avoid destroying pretrained features, then carefully fine-tunes higher-level convolutional filters.

**Inference thresholds & mapping**

For binary (sigmoid) models we apply thresholds used in the service (`model_service.py`):

- `ALS_THRESHOLD = 0.96` → sigmoid ≥ 0.96 ⇒ predict `angular_leaf_spot`
- `HEALTHY_THRESHOLD = 0.88` → sigmoid ≤ 0.88 ⇒ predict `healthy`
- `0.88 < sigmoid < 0.96` ⇒ `other_disease` (uncertain band)

For multi-class models (softmax), the predicted class is `argmax(p)` and the confidence is the max softmax probability.

Class mapping example (multi-class run with 4 folders sorted A→Z by Keras):

- Class 0 = `angular_leaf_spot`
- Class 1 = `bean_rust`
- Class 2 = `healthy`
- Class 3 = `other_leaves`

Confirm mapping by printing `train_raw.class_names` during dataset loading.

**Practical notes for a thesis**

- Reproducibility: fix random seeds (`SEED = 42`), and use deterministic dataset splits (Keras `validation_split` + `seed`).
- Explain augmentation rationale: improves generalization by simulating viewpoint/lighting variation.
- Show performance metrics: report validation accuracy and, for binary models, ROC AUC and confusion matrix. Mention class weighting when classes are imbalanced and show the class weight formula (above).
- If you include training curves in the thesis, plot loss and accuracy for both stages (head training and fine-tuning) and annotate where you unfreeze the base.

**Reproduce locally**

1. Ensure dependencies: `tensorflow`, `numpy`, `opencv-python`, `pillow` (same versions used during experiments).
2. Prepare `data/` folders with images organized by class.
3. Run: `python train_model.py`

**Where to look in the code**

- Training logic & parameters: `train_model.py`
- Dataset loading & augmentation: `train_model.py::load_datasets()`
- Thresholds and inference mapping: `model_service.py` (constants near top: `ALS_THRESHOLD`, `HEALTHY_THRESHOLD`)

If you'd like, I can also add a small script to export training logs (loss/accuracy) to CSV and generate plots for inclusion in your thesis figures.

---
Last updated: automatic from training scripts in this repository.
