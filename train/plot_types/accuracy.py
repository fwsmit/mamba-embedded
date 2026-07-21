import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig


def _has_quant16(data):
    """Return True if any entry has a valid quantized_accuracy_int16."""
    return any(not np.isnan(d.get("test_quantized_accuracy_int16", np.nan)) for d in data)


def _bar_groups(data):
    """Yield (values, kwargs) tuples for each bar group.

    Yield order: float, (int16 if available), int8, (mcu if available).
    Placing int16 before int8 keeps it adjacent to float for easy comparison.
    """
    # Always present
    yield [d["test_float_accuracy"] for d in data], {
        "label": "Full model (f32)", "color": "#4C9BE8",
    }

    # Optional int16 — before int8 so it sits next to float
    if _has_quant16(data):
        vals = []
        for d in data:
            v = d.get("test_quantized_accuracy_int16", np.nan)
            vals.append(v if not np.isnan(v) else 0.0)
        yield vals, {
            "label": "Quantized (int16)", "color": "#8E44AD",
        }

    # Always present
    yield [d["test_quantized_accuracy"] for d in data], {
        "label": "Quantized (int8)", "color": "#E8834C",
    }

    # Optional MCU
    if any(not np.isnan(d.get("mcu_accuracy", np.nan)) for d in data):
        vals = []
        for d in data:
            v = d.get("mcu_accuracy", np.nan)
            vals.append(v if not np.isnan(v) else 0.0)
        yield vals, {
            "label": "MCU Accuracy", "color": "#4CAF50",
        }


def create_accuracy_comparison_plot(study_name, data, title, show_mcu=False):
    """
    Plot float_accuracy vs quantized_accuracy for all models in a results.json.
    Also plots mcu_accuracy if available (on-device inference accuracy) and
    ``show_mcu`` is True.  When the data contains ``quantized_accuracy_int16``,
    a separate bar for 16-bit quantized accuracy is included.

    Parameters
    ----------
    study_name : str
        Name of the study (used for figure title and filename).
    data : list of dict
        Entries from results.json with float_accuracy and quantized_accuracy.
    title : str
        Used in the plot title and saved file names.
    show_mcu : bool
        If True, include MCU accuracy bars when MCU data exists.
    """
    # Filter out entries with NaN float_accuracy
    data = [d for d in data if not np.isnan(d.get("test_float_accuracy", np.nan))]
    data.sort(key=lambda d: d["test_float_accuracy"], reverse=True)

    if not data:
        print(f"  No valid accuracy entries found for {study_name}.")
        return

    # Build bar groups — optionally include MCU only if show_mcu is True
    groups = []
    for vals, kwargs in _bar_groups(data):
        lbl = kwargs["label"]
        if lbl == "MCU Accuracy" and not show_mcu:
            continue
        groups.append((vals, kwargs))

    n_groups = len(groups)
    trial_labels = [str(d["trial_number"]) for d in data]
    x = np.arange(len(data))

    # Figure size scales slightly with number of groups
    fig_width = 10 + max(0, n_groups - 2) * 1.5
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Bar width and offsets: cluster centered around each tick
    width = {2: 0.35, 3: 0.25, 4: 0.20}.get(n_groups, 0.20)
    all_bars = []
    for i, (vals, kwargs) in enumerate(groups):
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=kwargs["label"], color=kwargs["color"],
                      edgecolor="white")
        all_bars.append(bars)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Trial (sorted by float accuracy)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate bars with the accuracy value
    for bars in all_bars:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=6)

    savefig(fig, title, "accuracy")
