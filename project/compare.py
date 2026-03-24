"""
compare.py
──────────────────────────────────────────────────────────────────────────────
Reads the *_metrics.json files saved by evaluate.py and prints a clean
comparison table to the terminal.

Usage:
    python compare.py

Pre-requisite: run evaluate.py first.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json

from tabulate import tabulate


OUTPUTS_DIR  = "outputs"
MODEL_NAMES  = ["CNN", "ResNet18", "MobileNetV2"]


def load_metrics(model_name: str) -> dict | None:
    path = os.path.join(OUTPUTS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(path):
        print(f"  [WARN] Metrics file not found for '{model_name}': {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("\n" + "="*65)
    print("  WCE Colon Disease — Model Comparison")
    print("="*65)

    rows = []
    for name in MODEL_NAMES:
        metrics = load_metrics(name)
        if metrics is None:
            continue
        rows.append([
            name,
            f"{metrics['accuracy']:.2f}%",
            f"{metrics['precision']:.2f}%",
            f"{metrics['recall']:.2f}%",
            f"{metrics['f1']:.2f}%",
        ])

    if not rows:
        print("\n  No metrics found. Please run evaluate.py first.\n")
        return

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    # ── Highlight best model ──────────────────────────────────────────────────
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i][4].rstrip("%")))
    print(f"\n  🏆  Best model by F1-Score: {rows[best_idx][0]}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
