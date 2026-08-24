import numpy as np
import matplotlib.pyplot as plt

from .common import savefig, beeswarm_box

# Variant short names (as used in the x tick labels) in the canonical
# single → bidir add → bidir mul order, matching the other plot types.
VARIANTS = ["single", "bidir-add", "bidir-mul"]

# Marker per variant: distinct shapes survive grayscale printing.
VARIANT_MARKERS = {"single": "o", "bidir-add": "s", "bidir-mul": "^"}

# One colour per dataset, drawn from the shared COLORS palette.
DATASET_COLORS = {"har": "#4C9BE8", "kws": "#E8834C"}

# Dataset ordering on the x axis (HAR first, as in the other figures).
DATASET_ORDER = {"har": 0, "kws": 1}


def _dataset_from_study(study_name):
    """Dataset token of a study name like 'mamba-1-har-bidir' → 'har'."""
    parts = study_name.split("-")
    return parts[2] if len(parts) > 2 else study_name


def _variant_from_display(display_name):
    """Short variant name derived from the config's plot_description."""
    low = display_name.lower()
    if "add" in low:
        return "bidir-add"
    if "mul" in low:
        return "bidir-mul"
    return "single"


def create_val_test_gap_plot(studies_data, title, ylim=None):
    """
    Overfitting check: one beeswarm + box column per experiment showing the
    float model's validation-minus-test accuracy gap in percentage points.

    Positive values mean the validation accuracy exceeds the test accuracy,
    i.e. the model generalises less well to the held-out test partition. A
    dashed zero line is the "no gap" reference. Columns are ordered by dataset
    (HAR first) and then variant (single → bidir add → bidir mul), mirroring
    the quantization_loss figure's beeswarm + box + mean-diamond style.

    Parameters
    ----------
    studies_data : list of dict
        Studies with 'name' (display label), 'study_name' and 'results_data'
        (list of entries from results.json).
    title : str
        Basis for the saved file name.
    ylim : tuple of float or None
        Optional (y_low, y_high) applied to the y-axis.
    """
    cols = []  # (label, gaps, color, marker)
    for sd in studies_data:
        data = sd.get("results_data", [])
        if not data:
            continue
        dataset = _dataset_from_study(sd["study_name"])
        variant = _variant_from_display(sd["name"])
        gaps = []
        for rd in data:
            fa = rd.get("float_accuracy", np.nan)
            ta = rd.get("test_float_accuracy", np.nan)
            if not np.isnan(fa) and not np.isnan(ta):
                gaps.append(fa - ta)
        if not gaps:
            continue
        cols.append((f"{dataset}-{variant}", gaps,
                     DATASET_COLORS.get(dataset, "#4C9BE8"),
                     VARIANT_MARKERS[variant]))

    if not cols:
        print("  No float val/test accuracy data found across any study.")
        exit(1)

    # Sort by dataset (HAR first), then by variant order.
    cols.sort(key=lambda c: (DATASET_ORDER.get(c[0].split("-")[0], 99),
                             VARIANTS.index(c[0].split("-", 1)[1])))

    fig, ax = plt.subplots(figsize=(5.6, 3.3))

    # ── Raw points, box, mean diamond per experiment ────────────────────────
    for x, (label, gaps, color, marker) in enumerate(cols):
        beeswarm_box(ax, fig, x, gaps, color, marker)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Reference line: points below it have test accuracy above validation.
    ax.axhline(0, color="#7F7F7F", linestyle="--", linewidth=0.8, zorder=3)

    # ── Axes / labels ───────────────────────────────────────────────────────
    tick_labels = [f"{label}\n(n={len(gaps)})" for label, gaps, _c, _m in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Float validation - test accuracy (%pt)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    savefig(fig, title, "val_test_gap", dpi=300, svg=True)