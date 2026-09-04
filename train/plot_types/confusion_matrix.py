"""Confusion matrix of a specific trial's MCU predictions.

Reads the trial's per-class predictions produced by the firmware on the
validation set (`experiments/<study>/predictions_trial_<N>.json`) and the
matching ground truth (validation labels + class names, loaded the same way
the quantization pipeline builds the validation set). Plots a row-normalised
confusion matrix with per-cell counts.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common import savefig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# UCI HAR activity names; index == label (load_har_data subtracts 1 from the
# 1..6 labels in y_train.txt).
HAR_CLASSES = [
    "WALKING",            # 0
    "WALKING_UPSTAIRS",   # 1
    "WALKING_DOWNSTAIRS", # 2
    "SITTING",            # 3
    "STANDING",           # 4
    "LAYING",             # 5
]

# KWS 12-class label space (from kws_dataset_gen.ALL_LABELS).
KWS_CLASSES = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]


def infer_dataset(study_name: str):
    """Extract dataset name from a study name (e.g. 'mamba-1-kws-2' -> 'kws')."""
    for part in study_name.split("-"):
        if part in ("har", "kws"):
            return part
    return None


def _har_val_labels(data_root: Path):
    """Validation labels for HAR, replicating load_har_data()'s exact 80/20
    seed-42 random split (the split that produced the MCU dataset)."""
    import torch
    from torch.utils.data import TensorDataset, random_split

    har_dir = data_root / "har-uci-dataset" / "UCI HAR Dataset"
    if not (har_dir / "train" / "y_train.txt").exists():
        return None, None, str(har_dir)

    y_train = np.loadtxt(har_dir / "train" / "y_train.txt", dtype=np.int64).squeeze() - 1
    dummy = torch.zeros((len(y_train), 2))
    _, val_ds = random_split(
        TensorDataset(dummy, torch.zeros(len(y_train), dtype=torch.long)),
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(42),
    )
    return y_train[np.array(val_ds.indices)], HAR_CLASSES, None


def _kws_val(data_root: Path):
    """Validation labels + class names for KWS from the first val pkl found.

    Tries the canonical pipeline location (~/Datasets/speech_commands_v0.02_
    augmented/val.pkl) first, then the repo-local working copy
    (kws_dataset/val.pkl).
    """
    candidates = [
        data_root / "speech_commands_v0.02_augmented" / "val.pkl",
        REPO_ROOT / "kws_dataset" / "val.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            labels = np.asarray(d["y"])
            idx2label = d.get("idx2label")
            names = ([idx2label[i] for i in range(len(idx2label))]
                     if idx2label else KWS_CLASSES)
            return labels, names, None
    return None, None, str(candidates[0])


def create_confusion_matrix_plot(study_name: str, trial_number: int, title: str):
    """Confusion matrix of trial *trial_number*'s MCU predictions on the
    validation set against ground truth.

    Returns True if the figure was saved.
    """
    dataset = infer_dataset(study_name)
    if dataset is None:
        print(f"  Error: could not infer dataset from study name '{study_name}'")
        return False

    preds_path = REPO_ROOT / "experiments" / study_name / f"predictions_trial_{trial_number}.json"
    if not preds_path.exists():
        print(f"  Error: no predictions file at {preds_path}")
        return False
    with open(preds_path) as f:
        preds = np.asarray(json.load(f))

    data_root = Path.home() / "Datasets"
    if dataset == "har":
        labels, class_names, err_hint = _har_val_labels(data_root)
        hint = (f"raw UCI HAR dataset under {err_hint or data_root / 'har-uci-dataset'}")
    else:
        labels, class_names, err_hint = _kws_val(data_root)
        hint = f"val.pkl under {err_hint or data_root / 'speech_commands_v0.02_augmented'}"

    if labels is None:
        print(f"  Error: could not load {dataset} validation labels ({hint})")
        return False

    if len(preds) != len(labels):
        print(f"  Error: prediction count ({len(preds)}) doesn't match "
              f"validation label count ({len(labels)})")
        return False

    n_classes = len(class_names)
    bad = (preds < 0) | (preds >= n_classes)
    if bad.any():
        print(f"  Warning: {int(bad.sum())} predictions outside [0, {n_classes}), clipping")
        preds = np.clip(preds, 0, n_classes - 1)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    accuracy = np.trace(cm) / max(cm.sum(), 1) * 100.0

    fig, ax = plt.subplots(
        figsize=(max(6.5, n_classes * 0.72), max(5.5, n_classes * 0.62))
    )

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label (ESP32-S3)", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    # White grid lines between cells.
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    small = n_classes > 8
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            if small and cm[i, j] == 0:
                continue
            txt = (f"{cm[i, j]}" if small
                   else f"{cm[i, j]}\n({cm_norm[i, j]:.2f})")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7 if small else 8,
                    color="white" if cm_norm[i, j] > thresh else "black")

    ax.text(0.99, 0.99, f"Accuracy: {accuracy:.1f}%",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#BBBBBB", alpha=0.9))

    savefig(fig, title, "confusion_matrix", dpi=300)
    return True