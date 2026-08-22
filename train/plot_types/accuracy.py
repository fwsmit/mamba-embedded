import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig


def _has_quant16(data):
    """Return True if any entry has a valid quantized_accuracy_int16."""
    return any(not np.isnan(d.get("test_quantized_accuracy_int16", np.nan)) for d in data)


def _has_quant_strat(data):
    """Return True if any entry has a valid test_quantized_accuracy_strat."""
    return any(not np.isnan(d.get("test_quantized_accuracy_strat", np.nan)) for d in data)


def create_accuracy_comparison_plot(study_name, data, title):
    """
    Scatter plot of quantized accuracy (y) vs float accuracy (x) for all
    models in a results.json.  Each quantization method (int8, and int16 /
    strat-kl-tqt when present) is shown as a differently-coloured set of
    points.  A dotted y=x reference line is drawn for comparison.

    Parameters
    ----------
    study_name : str
        Name of the study (used for figure title and filename).
    data : list of dict
        Entries from results.json with float_accuracy and quantized_accuracy.
    title : str
        Used in the plot title and saved file names.
    """
    data = [d for d in data if not np.isnan(d.get("test_float_accuracy", np.nan))]
    data = [d for d in data if not np.isnan(d.get("test_quantized_accuracy", np.nan))]

    if not data:
        print(f"  No valid accuracy entries found for {study_name}.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # y = x reference line
    lo = min(
        min(d["test_float_accuracy"] for d in data),
        min(d["test_quantized_accuracy"] for d in data),
    )
    hi = max(
        max(d["test_float_accuracy"] for d in data),
        max(d["test_quantized_accuracy"] for d in data),
    )
    pad = max(1.0, (hi - lo) * 0.05)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="grey", linewidth=1.2)

    # Quantized (int8)
    ax.scatter([d["test_float_accuracy"] for d in data],
               [d["test_quantized_accuracy"] for d in data],
               s=45, color="#E8834C", edgecolor="white", linewidth=0.5,
               label="Quantized (int8)")

    # Optional quantized (int16)
    if _has_quant16(data):
        x16 = [d["test_float_accuracy"] for d in data]
        y16 = [d.get("test_quantized_accuracy_int16", np.nan) for d in data]
        mask = [not np.isnan(v) for v in y16]
        ax.scatter([v for v, m in zip(x16, mask) if m],
                   [v for v, m in zip(y16, mask) if m],
                   s=45, color="#8E44AD", edgecolor="white", linewidth=0.5,
                   label="Quantized (int16)")

    # Optional strat-kl-tqt quantized
    if _has_quant_strat(data):
        xs = [d["test_float_accuracy"] for d in data]
        ys = [d.get("test_quantized_accuracy_strat", np.nan) for d in data]
        mask = [not np.isnan(v) for v in ys]
        ax.scatter([v for v, m in zip(xs, mask) if m],
                   [v for v, m in zip(ys, mask) if m],
                   s=45, color="#2E7D32", edgecolor="white", linewidth=0.5,
                   label="Quantized (strat-kl-tqt)")

    ax.set_xlabel("Float Accuracy (%)", fontsize=11)
    ax.set_ylabel("Quantized Accuracy (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    savefig(fig, title, "accuracy")


def create_accuracy_comparison_bar_plot(study_name, data, title):
    """
    Grouped bar chart of float vs quantized accuracy for all models in a
    results.json.  One bar group per trial, one bar per quantization method
    (float, int8, and int16 / strat-kl-tqt when present).

    Parameters
    ----------
    study_name : str
        Name of the study (used for figure title and filename).
    data : list of dict
        Entries from results.json with float_accuracy and quantized_accuracy.
    title : str
        Used in the plot title and saved file names.
    """
    data = [d for d in data if not np.isnan(d.get("test_float_accuracy", np.nan))]
    data = [d for d in data if not np.isnan(d.get("test_quantized_accuracy", np.nan))]

    if not data:
        print(f"  No valid accuracy entries found for {study_name}.")
        return

    data = sorted(data, key=lambda d: d.get("trial_number", 0), reverse=False)

    methods = [("Float", "test_float_accuracy", "#4C9BE8")]
    if _has_quant16(data):
        methods.append(("Quantized (int16-percent)", "test_quantized_accuracy_int16", "#8E44AD"))
    methods.append(("Quantized (int8-percent)", "test_quantized_accuracy", "#E8834C"))
    if _has_quant_strat(data):
        methods.append(("Quantized (int8-kl-tqt)", "test_quantized_accuracy_strat", "#2E7D32"))

    n = len(data)
    x = np.arange(n)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * n + 4), 6))

    hi = 0.0
    for i, (label, field, color) in enumerate(methods):
        vals = np.array([d.get(field, np.nan) for d in data], dtype=float)
        valid = vals[~np.isnan(vals)]
        if valid.size:
            hi = max(hi, float(valid.max()))
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=color,
               edgecolor="white", linewidth=0.5, label=label)

    ax.set_ylim(0, hi * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d.get("trial_number", i)) for i, d in enumerate(data)],
                       fontsize=9)
    ax.set_xlabel("Trial nr", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.legend(fontsize=10, loc="lower left", framealpha=1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    savefig(fig, title, "accuracy")
