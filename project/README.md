# WCE Curated Colon Disease Classification
### Deep Learning Project — Wireless Capsule Endoscopy Images

---

## 📁 Project Structure

```
project/
│── data/               ← Place dataset here (train/ val/ test/ subfolders)
│── models/             ← Saved model weights (.pth)
│── outputs/            ← Plots, metrics, confusion matrices
│── utils/
│   └── dataset.py      ← Dataset loading & preprocessing
│── models/
│   ├── __init__.py
│   ├── cnn.py          ← Custom SimpleCNN
│   ├── resnet.py       ← Pretrained ResNet-18
│   └── mobilenet.py    ← Pretrained MobileNetV2
│── train.py            ← Train all 3 models
│── evaluate.py         ← Evaluate on test set
│── compare.py          ← Print comparison table
│── requirements.txt
│── README.md
```

---

## 🗂️ Dataset Setup

The dataset (WCE Curated Colon Disease) should be placed inside `data/`.

**Option A — Pre-split (recommended, matches Kaggle download):**
```
data/
  train/
    class_A/  img1.jpg ...
    class_B/  img2.jpg ...
  val/
    class_A/  ...
    class_B/  ...
  test/
    class_A/  ...
    class_B/  ...
```

**Option B — Single root folder:**
```
data/
  class_A/  img1.jpg ...
  class_B/  img2.jpg ...
```
The code will automatically perform a **70/15/15** train/val/test split.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+. CUDA is auto-detected.

---

## 🚀 How to Run (Step by Step)

### Step 1 — Train all models
```bash
python train.py
```
- Trains **SimpleCNN**, **ResNet-18**, and **MobileNetV2**
- Saves best weights → `models/<ModelName>_best.pth`
- Saves training curves → `outputs/<ModelName>_training_curve.png`

### Step 2 — Evaluate on test set
```bash
python evaluate.py
```
- Loads saved weights
- Computes Accuracy, Precision, Recall, F1-Score
- Prints per-class classification report
- Saves confusion matrices → `outputs/<ModelName>_confusion_matrix.png`

### Step 3 — Compare all models
```bash
python compare.py
```
- Reads saved metric files from `outputs/`
- Prints a formatted comparison table

---

## 📊 Models

| Model        | Type              | Params (approx) |
|:-------------|:------------------|:----------------|
| SimpleCNN    | Custom (4 blocks) | ~10M            |
| ResNet-18    | Pretrained        | ~11M            |
| MobileNetV2  | Pretrained (lite) | ~3.4M           |

---

## 🔧 Hyperparameters

| Setting     | Value          |
|:------------|:---------------|
| Image size  | 224 × 224      |
| Batch size  | 32             |
| Epochs      | 10             |
| Optimizer   | Adam (lr=0.001)|
| Loss        | CrossEntropy   |
| Scheduler   | ReduceLROnPlateau |

---

## 📤 Outputs

| File                                    | Description                    |
|:----------------------------------------|:-------------------------------|
| `models/<Name>_best.pth`               | Best model weights             |
| `outputs/<Name>_training_curve.png`    | Accuracy & loss curves         |
| `outputs/<Name>_confusion_matrix.png`  | Confusion matrix heatmap       |
| `outputs/<Name>_metrics.json`          | Metrics (used by compare.py)   |
| `outputs/class_names.json`             | Class label list               |

---

## 💡 Notes

- GPU (CUDA) is used automatically if available.
- `NUM_WORKERS = 0` is set for Windows compatibility.
- To adjust epochs/batch size, edit the constants at the top of `train.py`.
