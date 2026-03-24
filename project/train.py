"""
train.py
──────────────────────────────────────────────────────────────────────────────
Train 3 models on the WCE Colon Disease dataset:
  1. SimpleCNN   (custom convolutional network)
  2. ResNet-18   (pretrained ImageNet, fine-tuned)
  3. MobileNetV2 (pretrained ImageNet, fine-tuned)

Usage:
    python train.py

Trained weights are saved to models/<model_name>_best.pth
Training curves are saved to outputs/<model_name>_training_curve.png
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from models import SimpleCNN, get_resnet18, get_mobilenetv2
from utils.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "../data"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.001
NUM_WORKERS = 0  # Set to 0 for Windows compatibility (avoids multiprocessing issues)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
    return device


def save_curve(history: dict, model_name: str) -> None:
    """Save training vs validation accuracy + loss curves."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[0].plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    axes[0].set_title(f"{model_name} – Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[1].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[1].set_title(f"{model_name} – Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"{model_name}_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Chart] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
) -> dict:
    """
    Train a single model for EPOCHS epochs, saving the best checkpoint.

    Returns:
        history dict with train/val loss and accuracy per epoch.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Learning-rate scheduler: reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_val_acc = 0.0
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    os.makedirs(MODELS_DIR, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]  ", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        # ── Scheduler & checkpoint ─────────────────────────────
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # ── Log ────────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    print(f"\n  Best Val Acc: {best_val_acc:.2f}%  →  Saved to {save_path}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(class_names)
    print(f"\n[Info] Number of classes: {num_classes}")

    device = get_device()

    # ── Model registry ────────────────────────────────────────
    models_to_train = {
        "CNN":         SimpleCNN(num_classes),
        "ResNet18":    get_resnet18(num_classes),
        # "MobileNetV2": get_mobilenetv2(num_classes),  # already trained — skip
    }

    all_histories = {}

    for name, model in models_to_train.items():
        history = train_model(model, name, train_loader, val_loader, device)
        save_curve(history, name)
        all_histories[name] = history

    # ── Save class names for use in evaluate.py ───────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"\n[Info] Class names saved → {OUTPUTS_DIR}/class_names.json")

    print("\n" + "="*60)
    print("  All models trained successfully!")
    print("  Run:  python evaluate.py   to get detailed metrics.")
    print("  Run:  python compare.py    to see a comparison table.")
    print("="*60)


if __name__ == "__main__":
    main()
