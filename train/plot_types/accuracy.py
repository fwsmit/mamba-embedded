import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig


def create_accuracy_comparison_plot(study_name, data, title, show_mcu=False):
    """
    Plot float_accuracy vs quantized_accuracy for all models in a results.json.
    Also plots mcu_accuracy if available (on-device inference accuracy) and
    ``show_mcu`` is True.

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
    data = [d for d in data if not np.isnan(d.get("float_accuracy", np.nan))]
    data.sort(key=lambda d: d["quantized_accuracy"], reverse=True)

    if not data:
        print(f"  No valid accuracy entries found for {study_name}.")
        return

    has_mcu = show_mcu and any(not np.isnan(d.get("mcu_accuracy", np.nan)) for d in data)

    trial_labels = [str(d["trial_number"]) for d in data]
    x = np.arange(len(data))

    if has_mcu:
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))

        bars_float = ax.bar(x - width, [d["float_accuracy"] for d in data],
                            width, label="Float Accuracy", color="#4C9BE8", edgecolor="white")
        bars_quant = ax.bar(x, [d["quantized_accuracy"] for d in data],
                            width, label="Quantized Accuracy", color="#E8834C", edgecolor="white")

        mcu_vals = []
        for d in data:
            v = d.get("mcu_accuracy", np.nan)
            mcu_vals.append(v if not np.isnan(v) else 0.0)
        bars_mcu = ax.bar(x + width, mcu_vals,
                          width, label="MCU Accuracy", color="#4CAF50", edgecolor="white")

        title = f"{title}"
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))

        bars_float = ax.bar(x - width / 2, [d["float_accuracy"] for d in data],
                            width, label="Float Accuracy", color="#4C9BE8", edgecolor="white")
        bars_quant = ax.bar(x + width / 2, [d["quantized_accuracy"] for d in data],
                            width, label="Quantized Accuracy", color="#E8834C", edgecolor="white")
        bars_mcu = None

        title = f"{title}"

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Trial (sorted by quantized accuracy)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate bars with the accuracy value
    for bar in bars_float:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=6)
    for bar in bars_quant:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=6)
    if bars_mcu is not None:
        for bar in bars_mcu:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=6)

    savefig(fig, title, "accuracy")
