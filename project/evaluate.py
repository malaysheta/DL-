"""
evaluate.py
──────────────────────────────────────────────────────────────────────────────
Evaluate the MobileNetV2 model on the test set and compute:
  • Accuracy
  • Precision (macro)
  • Recall    (macro)
  • F1-Score  (macro)
  • Confusion Matrix (saved as PNG)

Usage:
    python evaluate.py

Pre-requisite: run train.py first to generate weights in models/
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from models import get_mobilenetv2
from utils.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Config  (must match train.py)
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "../data"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
BATCH_SIZE  = 32
NUM_WORKERS = 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    return device


def load_class_names() -> list:
    path = os.path.join(OUTPUTS_DIR, "class_names.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run train.py first."
        )
    with open(path) as f:
        return json.load(f)


def predict(model: torch.nn.Module, loader, device: torch.device):
    """Run inference and return (all_labels, all_preds) as numpy arrays."""
    model.eval()
    model.to(device)
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(labels, preds) -> dict:
    """Compute classification metrics and return as a dict."""
    return {
        "accuracy":  accuracy_score(labels, preds) * 100,
        "precision": precision_score(labels, preds, average="macro", zero_division=0) * 100,
        "recall":    recall_score(labels, preds, average="macro", zero_division=0) * 100,
        "f1":        f1_score(labels, preds, average="macro", zero_division=0) * 100,
    }


def plot_confusion_matrix(labels, preds, class_names: list, model_name: str) -> None:
    """Save confusion matrix heatmap to outputs/."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Chart] Confusion matrix saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, model_name: str, weight_path: str,
                   test_loader, device, class_names) -> dict:
    """Load weights, run evaluation, print report, save confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(weight_path):
        print(f"  [WARN] Weights not found at '{weight_path}'. Skipping.")
        return {}

    model.load_state_dict(torch.load(weight_path, map_location=device))
    labels, preds = predict(model, test_loader, device)

    metrics = compute_metrics(labels, preds)
    print(f"\n  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall   : {metrics['recall']:.2f}%")
    print(f"  F1-Score : {metrics['f1']:.2f}%")

    # ── Pretty per-class report WITHOUT support column ──────────────────────
    print(f"\n  Per-class Report:\n")
    report_dict = classification_report(
        labels, preds, target_names=class_names,
        zero_division=0, output_dict=True
    )
    col_w = max(len(n) for n in class_names + ["weighted avg"]) + 2
    header = f"  {'':>{col_w}}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}"
    sep    = "  " + "-" * (col_w + 38)
    print(header)
    print(sep)
    for name in class_names:
        r = report_dict[name]
        print(f"  {name:>{col_w}}  {r['precision']:>10.2f}  {r['recall']:>10.2f}  {r['f1-score']:>10.2f}")
    print(sep)
    for avg in ("macro avg", "weighted avg"):
        r = report_dict[avg]
        print(f"  {avg:>{col_w}}  {r['precision']:>10.2f}  {r['recall']:>10.2f}  {r['f1-score']:>10.2f}")
    print()

    plot_confusion_matrix(labels, preds, class_names, model_name)

    # Save metrics JSON for compare.py to read
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUTS_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    class_names = load_class_names()
    num_classes = len(class_names)
    device      = get_device()

    _, _, test_loader, _ = build_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    model_configs = [
        ("MobileNetV2", get_mobilenetv2(num_classes), os.path.join(MODELS_DIR, "MobileNetV2_best.pth")),
    ]

    all_metrics = {}
    for name, model, path in model_configs:
        metrics = evaluate_model(model, name, path, test_loader, device, class_names)
        if metrics:
            all_metrics[name] = metrics

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
