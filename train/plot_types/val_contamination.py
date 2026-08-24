"""Per-subject contamination of the HAR UCI validation split.

The project's ``load_har_data()`` partitions the UCI HAR **train** directory
(7352 windows, subject-major chronological order) into train / validation via
``random_split([0.8, 0.2], generator=torch.Generator().manual_seed(42))``.
Because the raw sensor recordings are 128-reading sliding windows with 50%
overlap, two *adjacent* windows of the same subject share 64 of 128 raw
readings. When such an adjacent pair lands on opposite sides of the train/val
boundary, the validation window is "contaminated": the model saw half of that
window's raw signal during training.

A window pair is only counted as actually overlapping when the two windows
share identical raw readings (same subject, all 9 inertial axes, window ``i``
last-64 == window ``i+1`` first-64). Pairs at trial boundaries (activity
change or a restarted recording) do NOT share data and are excluded, so the
contamination is never over-counted.

This module produces a bar chart showing, for each of the 30 UCI subjects,
what fraction of that subject's *validation* windows are contaminated by an
overlapping train window. Subjects held out to the UCI test partition (which
therefore have no validation windows here) are drawn grey to flag "no
validation data".
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

import matplotlib.pyplot as plt

from .common import savefig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Subjects whose windows were all assigned to the UCI *test* partition, so
# they have no validation windows in the project's random split.
TEST_ONLY_SUBJECTS = {2, 4, 9, 10, 12, 13, 18, 20, 24}


def _inertial(har_dir, fname):
    return np.loadtxt(
        os.path.join(har_dir, "train", "Inertial Signals", fname),
        dtype=np.float64,
    )


def _true_overlap(axes, i, j):
    """True if windows i and j share 64 of 128 raw readings on every axis."""
    return all(np.array_equal(axes[a][i, 64:], axes[a][j, :64]) for a in axes)


def _subjects_and_contamination(har_dir):
    """Return (subjects 1..30, val_overlap_count, val_total) arrays.

    Per subject:
      val_overlap_count = distinct validation windows overlapped by a train
                          window (50% shared raw data), counted once each;
      val_total         = number of that subject's validation windows.
    """
    axis_files = sorted(
        f for f in os.listdir(os.path.join(har_dir, "train", "Inertial Signals"))
        if f.endswith(".txt")
    )
    axes = {f: _inertial(har_dir, f) for f in axis_files}

    subj_train = np.loadtxt(os.path.join(har_dir, "train", "subject_train.txt"),
                            dtype=int)
    n = len(subj_train)

    # Replicate load_har_data()'s exact 80/20, seed-42 random split.
    dummy = torch.zeros((n, 2))
    tr_ds, va_ds = random_split(
        TensorDataset(dummy, torch.zeros(n, dtype=torch.long)),
        [0.8, 0.2], generator=torch.Generator().manual_seed(42),
    )
    split_of = np.full(n, -1)                        # 0 = train, 1 = val
    split_of[np.array(tr_ds.indices)] = 0
    split_of[np.array(va_ds.indices)] = 1

    # Distinct indices involved in a genuine leak (same subject, adjacent,
    # cross-split, actually overlapping 50% raw data).
    involved = set()
    for i in range(n - 1):
        if (subj_train[i] == subj_train[i + 1]
                and split_of[i] != split_of[i + 1]
                and _true_overlap(axes, i, i + 1)):
            involved.add(i)
            involved.add(i + 1)

    subjects = np.arange(1, 31)
    val_ov = np.zeros(30, dtype=int)
    val_tot = np.zeros(30, dtype=int)
    for k, s in enumerate(subjects):
        m = subj_train == s
        val_tot[k] = int((split_of[m] == 1).sum())
        val_idx = np.flatnonzero(m & (split_of == 1))
        val_ov[k] = sum(1 for j in val_idx if int(j) in involved)

    return subjects, val_ov, val_tot


def create_har_val_contamination_plot(title="HAR validation contamination per subject",
                                      data_root=None):
    """Bar chart of per-subject contamination percentage of the validation set.

    For each of the 30 UCI subjects, the bar height is the percentage of that
    subject's validation windows that are overlapped by a train window (50%
    shared raw data). Subjects held out to the UCI test partition have no
    validation windows here and are drawn hatched grey at 0%. A dashed
    reference line marks the dataset-wide mean contamination of subjects that
    have validation data.
    """
    if data_root is None:
        data_root = REPO_ROOT / "data"
    har_dir = os.path.join(data_root, "har-uci-dataset", "UCI HAR Dataset")

    subjects, val_ov, val_tot = _subjects_and_contamination(har_dir)
    pct = np.divide(100.0 * val_ov, val_tot, out=np.zeros_like(val_ov, dtype=float),
                    where=val_tot > 0)

    # Mean over subjects that actually have validation windows (the 21
    # train/val subjects); exclude the 9 test-only subjects (val_tot==0).
    has_val = val_tot > 0
    mean_pct = float(np.mean(pct[has_val]))

    x = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(11, 5))

    # Solid bars for contaminated subjects, hatched grey for those with no
    # validation data in this random split (UCI test holdouts).
    ax.bar(x[has_val], pct[has_val], color="#4C9BE8",
           edgecolor="white", linewidth=0.4, zorder=3)
    ax.bar(x[~has_val], pct[~has_val], color="#CCCCCC", hatch="//",
           edgecolor="#888888", linewidth=0.6, zorder=3)

    # Dataset-wide mean reference (subjects with validation data only).
    ax.axhline(mean_pct, color="#A84F1A", linestyle="--", linewidth=1.2,
               zorder=2)

    ax.set_ylabel("Validation windows contaminated (%)", fontsize=11)
    ax.set_xlabel("Subject", fontsize=11)
    ax.set_xticks(x)
    ax.set_xlim(0.5, 30.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    # Compact legend clarifying the hatched bars.
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor="#4C9BE8", edgecolor="white"),
         Patch(facecolor="#CCCCCC", hatch="//", edgecolor="#888888")],
        ["Validation data", "No validation data"],
        loc="upper right", fontsize=8, frameon=False,
    )

    filename = "val_contamination"
    savefig(fig, title, filename, dpi=300)
    return True