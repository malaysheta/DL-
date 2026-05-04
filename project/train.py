"""
train.py
──────────────────────────────────────────────────────────────────────────────
Train MobileNetV2 on the WCE Colon Disease dataset.

Anti-overfitting strategy:
  ✓ Pre-split data used as-is  (train / val / test — NO re-splitting)
  ✓ Base layers FROZEN          (only custom head is trained)
  ✓ Batch Normalization         (in the classification head)
  ✓ Dropout (0.5 + 0.3)        (in the classification head)
  ✓ Early Stopping (patience=4) (stops when val_loss stops improving)
  ✓ ReduceLROnPlateau           (halves LR if stuck)
  ✓ Weight Decay (L2)           (optimizer-level regularisation)
  ✓ 15 epochs max

Usage:
    cd d:/PlaceMent/DL/project
    python train.py

Outputs:
    models/MobileNetV2_best.pth          ← best checkpoint (by val accuracy)
    outputs/MobileNetV2_training_curve.png
    outputs/class_names.json
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from models import get_mobilenetv2
from utils.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR      = "../data"          # root that contains train/ val/ test/
MODELS_DIR    = "models"
OUTPUTS_DIR   = "outputs"

BATCH_SIZE    = 32
EPOCHS        = 15                 # maximum epochs (early stopping may exit sooner)
LR            = 1e-4               # lower LR suits frozen-backbone transfer learning
WEIGHT_DECAY  = 1e-4               # L2 regularisation
PATIENCE      = 4                  # early-stopping patience (epochs)
NUM_WORKERS   = 0                  # 0 = safe on Windows (avoids spawn issues)


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU : {torch.cuda.get_device_name(0)}")
    return device


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def save_curve(history: dict, model_name: str) -> None:
    """Save training vs validation accuracy + loss curves to outputs/."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    epochs_ran = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[0].plot(epochs_ran, history["train_acc"], "b-o", label="Train Acc")
    axes[0].plot(epochs_ran, history["val_acc"],   "r-o", label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[1].plot(epochs_ran, history["train_loss"], "b-o", label="Train Loss")
    axes[1].plot(epochs_ran, history["val_loss"],   "r-o", label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"{model_name}_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Chart] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop  (with Early Stopping)
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
) -> dict:
    """
    Train model for up to EPOCHS epochs with early stopping.

    Anti-overfitting mechanisms active:
        • Frozen base      (set in models/mobilenet.py)
        • BatchNorm + Drop (set in models/mobilenet.py)
        • Weight decay     (AdamW optimizer)
        • Early stopping   (patience = PATIENCE epochs on val_loss)
        • ReduceLROnPlateau (halve LR after 2 stagnant epochs)

    Returns:
        history dict with per-epoch train/val loss and accuracy.
    """
    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"  Max epochs : {EPOCHS}   |  Early-stop patience : {PATIENCE}")
    print(f"  LR : {LR}   |  Weight decay : {WEIGHT_DECAY}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer : only update parameters that require grad ──────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n  Trainable params : {sum(p.numel() for p in trainable_params):,}")
    print(f"  Frozen   params  : {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n")

    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ── LR scheduler ─────────────────────────────────────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ── Checkpoint & early-stop state ─────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss":  [],
        "train_acc":  [], "val_acc":   [],
    }

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
            leave=False,
        )
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]  ",
                leave=False,
            )
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, predicted  = outputs.max(1)
                val_correct  += predicted.eq(labels).sum().item()
                val_total    += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        # ── Scheduler step (on val_loss) ──────────────────────────────────────
        scheduler.step(val_loss)

        # ── Checkpoint (save best by val_loss) ───────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            ckpt_flag = "  ✓ saved"
        else:
            patience_counter += 1
            ckpt_flag = f"  (patience {patience_counter}/{PATIENCE})"

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s{ckpt_flag}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Early stopping check ──────────────────────────────────────────────
        if patience_counter >= PATIENCE:
            print(f"\n  [Early Stop] Val loss did not improve for {PATIENCE} epochs.")
            print(f"  [Early Stop] Stopped at epoch {epoch}.")
            break

    print(f"\n  Best Val Loss : {best_val_loss:.4f}  |  Best Val Acc : {best_val_acc:.2f}%")
    print(f"  Best checkpoint → {save_path}")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  MobileNetV2 — Anti-Overfitting Training Pipeline")
    print("  Strategy: Frozen base | BN | Dropout | Early Stopping")
    print("="*60)

    # ── Data  (pre-split, no re-splitting) ────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(class_names)
    print(f"\n[Info] Classes ({num_classes}): {class_names}")

    device = get_device()

    # ── Build model ───────────────────────────────────────────────────────────
    model = get_mobilenetv2(num_classes)

    # ── Train ─────────────────────────────────────────────────────────────────
    history = train_model(model, "MobileNetV2", train_loader, val_loader, device)
    save_curve(history, "MobileNetV2")

    # ── Persist class names for evaluate.py ───────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cls_path = os.path.join(OUTPUTS_DIR, "class_names.json")
    with open(cls_path, "w") as f:
        json.dump(class_names, f)
    print(f"\n[Info] Class names saved → {cls_path}")

    print("\n" + "="*60)
    print("  Training complete!")
    print("  Next steps:")
    print("    python evaluate.py   ← detailed metrics on test set")
    print("    python compare.py    ← comparison table")
    print("="*60)


if __name__ == "__main__":
    main()
