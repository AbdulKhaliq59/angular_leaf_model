# Angular Leaf Spot Detection — ML Model & API

A Flask REST API that detects Angular Leaf Spot (ALS) and Bean Rust on bean leaves using a MobileNetV2 transfer-learning model trained on 4 classes.

## Detection Classes

| Class | Meaning |
|---|---|
| `angular_leaf_spot` | Bean leaf with ALS disease |
| `bean_rust` | Bean leaf with Bean Rust disease |
| `healthy` | Healthy bean leaf |
| `other_leaves` | Not a bean leaf (rejected) |

The 4-stage inference pipeline also rejects images before they reach the model:
- **Stage 1** — Blur / size check → `low_quality`
- **Stage 2** — HSV leaf-colour check → `not_leaf`
- **Stage 3** — Contour shape check → `not_bean_leaf`
- **Stage 4** — Model inference → one of the 4 classes above

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | **3.10 or 3.12** (3.9 crashes on Apple Silicon with TF 2.20+) |
| pip | any recent version |

---

## Local Setup

### macOS (Intel & Apple Silicon)

```bash
# 1. Clone / navigate to this folder
cd angular_leaf_model

# 2. Create a virtual environment with the correct Python
#    Apple Silicon (M1/M2/M3):
/opt/homebrew/bin/python3.12 -m venv venv312
source venv312/bin/activate

#    Intel Mac (system Python 3.10+ is usually fine):
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install datasets library for downloading training data
pip install datasets pillow
```

### Windows

```bat
REM 1. Open Command Prompt or PowerShell and navigate to the folder
cd angular_leaf_model

REM 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

REM 3. Install dependencies
pip install -r requirements.txt

REM 4. (Optional) Install datasets library for downloading training data
pip install datasets pillow
```

> **Windows GPU note:** If you have an NVIDIA GPU, install the CUDA-enabled wheel instead:
> `pip install tensorflow[and-cuda]`

---

## Training Your Own Model

### Step 1 — Prepare data folders

The training script auto-detects however many class folders you create inside `data/`:

```
data/
├── angular_leaf_spot/   ← place diseased bean leaf images here
├── bean_rust/           ← place bean rust images here
├── healthy/             ← place healthy bean leaf images here
└── other_leaves/        ← place non-bean leaf images here
```

### Step 2 — Download `other_leaves` images (iNaturalist, no auth)

```bash
# macOS / Apple Silicon
python -u prepare_other_leaves.py --count 300

# Windows
python prepare_other_leaves.py --count 300
```

### Step 3 — Download `bean_rust` images (Hugging Face, no auth)

```bash
# macOS / Apple Silicon
python -u prepare_bean_diseases.py

# Windows
python prepare_bean_diseases.py
```

This downloads ~436 Bean Rust images from the public `beans` dataset on Hugging Face.

### Step 4 — Train the model

```bash
# macOS — use the venv that has TensorFlow installed
PYTHONUNBUFFERED=1 ./venv312/bin/python -u train_model.py

# Windows
python train_model.py
```

Training runs two stages:
1. **Head only** (15 epochs) — MobileNetV2 base frozen
2. **Fine-tune** (25 epochs) — top 30 layers of MobileNetV2 unfrozen

The best checkpoint is saved to `models/beenleaf_model.h5` (~25 MB).

Expected class layout after training (sorted alphabetically by Keras):

| Index | Folder | Outcome |
|---|---|---|
| 0 | `angular_leaf_spot` | ALS_DETECTED |
| 1 | `bean_rust` | NON_ALS_DISEASE |
| 2 | `healthy` | HEALTHY_BEAN_LEAF |
| 3 | `other_leaves` | NON_BEAN_LEAF (rejected) |

---

## Running the API Server

```bash
# macOS
./venv312/bin/python api_server.py

# Windows
python api_server.py
```

The server starts at **http://localhost:5001**.

### Production (Gunicorn — macOS/Linux only)

```bash
./venv312/bin/gunicorn -w 2 -b 0.0.0.0:5001 api_server:app
```

---

## API Endpoints

### `GET /`  Health check

```json
{
  "status": "healthy",
  "service": "Angular Leaf Spot Detection API",
  "model_loaded": true
}
```

### `POST /predict`  Single image

```bash
curl -X POST http://localhost:5001/predict \
  -F "image=@path/to/leaf.jpg"
```

**Success response:**
```json
{
  "success": true,
  "prediction": {
    "predicted_class": "angular_leaf_spot",
    "status": "unhealthy",
    "health_status": "UNHEALTHY",
    "is_leaf": true,
    "confidence": 0.9741,
    "result": "Angular Leaf Spot Detected",
    "interpretation": "softmax[als]=0.974 → ALS"
  }
}
```

**Rejection example (blurry image):**
```json
{
  "success": true,
  "prediction": {
    "predicted_class": "low_quality",
    "status": "rejected",
    "health_status": "LOW_QUALITY",
    "is_leaf": false,
    "confidence": 0.0,
    "result": "Low Quality Image",
    "interpretation": "Image quality insufficient: image too blurry (score=12.3)."
  }
}
```

### `POST /predict/batch`  Multiple images

```bash
curl -X POST http://localhost:5001/predict/batch \
  -F "images=@leaf1.jpg" \
  -F "images=@leaf2.jpg"
```

---

## File Structure

```
angular_leaf_model/
├── api_server.py              # Flask API entry point
├── model_service.py           # 4-stage inference pipeline
├── train_model.py             # MobileNetV2 training script
├── prepare_other_leaves.py    # Download other-leaf images (iNaturalist)
├── prepare_bean_diseases.py   # Download bean-rust images (Hugging Face)
├── requirements.txt
├── models/
│   └── beenleaf_model.h5      # Trained model (not in git)
└── data/                      # Training images (not in git)
    ├── angular_leaf_spot/
    ├── bean_rust/
    ├── healthy/
    └── other_leaves/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `abort` / `mutex lock failed` on Mac | Use Python 3.12 via Homebrew: `/opt/homebrew/bin/python3.12` |
| `ModuleNotFoundError: numpy` | Activate the venv before running: `source venv312/bin/activate` |
| `Model not found` | Run `train_model.py` first, or place `beenleaf_model.h5` in `models/` |
| `expected shape=(None,224,224,3)` | You're using an old model; retrain with current `train_model.py` |
| Port 5001 already in use | `PORT=5002 python api_server.py` |
| `iNaturalist API hanging` | Already fixed — script uses `order=desc&order_by=id` |
