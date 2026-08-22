import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .common import savefig


def _clean(data):
    """Filter to entries with valid float and int8 test accuracies."""
    data = [d for d in data if not np.isnan(d.get("test_float_accuracy", np.nan))]
    data = [d for d in data if not np.isnan(d.get("test_quantized_accuracy", np.nan))]
    return data


def _has_quant16(data):
    """Return True if any entry has a valid quantized_accuracy_int16."""
    return any(not np.isnan(d.get("test_quantized_accuracy_int16", np.nan)) for d in data)


def create_accuracy_grid_plot(studies_data, title, ncols=2):
    """
    Combined accuracy scatter plot: one panel per study in a shared grid.

    Studies are grouped by column: each contiguous block of ``nrows`` studies
    is placed in its own column (so the same model-type order stacks vertically
    per dataset). Every panel shares axis limits and a y=x reference line, and
    a single shared legend is drawn in the first panel. Tick numbers and axis
    labels are shown only on the outer edges to keep the figure compact.

    Parameters
    ----------
    studies_data : list of dict
        Each entry has ``name`` (display name), ``results_data`` and ``color``.
    title : str
        Common figure title (also used for the output filename).
    ncols : int
        Number of columns in the subplot grid (each column holds a group).
    """
    valid = [sd for sd in studies_data if _clean(sd["results_data"])]
    if not valid:
        print("  No valid accuracy entries found for any study.")
        return

    # Shared axis limits over ALL panels so they are comparable.
    floats = [d["test_float_accuracy"] for sd in valid for d in _clean(sd["results_data"])]
    quants = [d["test_quantized_accuracy"] for sd in valid for d in _clean(sd["results_data"])]
    lo = min(min(floats), min(quants))
    hi = max(max(floats), max(quants))
    pad = max(1.0, (hi - lo) * 0.05)
    lo -= pad
    hi += pad

    n = len(valid)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6.0 * ncols, 3.0 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.92, bottom=0.12)

    legend_ready = False
    for i, sd in enumerate(valid):
        row = i % nrows          # fill down each column first → group by column
        col = i // nrows
        data = _clean(sd["results_data"])
        ax = fig.add_subplot(gs[row, col])

        # y = x reference line
        ax.plot([lo, hi], [lo, hi], linestyle=":", color="grey", linewidth=1.2)

        # Quantized (int8)
        sc = ax.scatter([d["test_float_accuracy"] for d in data],
                        [d["test_quantized_accuracy"] for d in data],
                        s=40, color=sd["color"], edgecolor="white", linewidth=0.5,
                        label="Quantized (int8)")

        # Optional quantized (int16)
        if _has_quant16(data):
            y16 = [d.get("test_quantized_accuracy_int16", np.nan) for d in data]
            mask = [not np.isnan(v) for v in y16]
            x16 = [d["test_float_accuracy"] for d in data]
            sc16 = ax.scatter([v for v, m in zip(x16, mask) if m],
                              [v for v, m in zip(y16, mask) if m],
                              s=40, color="#8E44AD", edgecolor="white", linewidth=0.5,
                              label="Quantized (int16)")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

        # Add dataset indicator
        if "kws" in sd["name"].lower():
            ax.text(0.05, 0.95, "KWS", transform=ax.transAxes, fontsize=10,
                    fontweight='bold', color='darkred', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        elif "har" in sd["name"].lower():
            ax.text(0.05, 0.95, "HAR", transform=ax.transAxes, fontsize=10,
                    fontweight='bold', color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        if col == 0:
            ax.set_ylabel("Quantized Accuracy (%)", fontsize=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if row == nrows - 1:
            ax.set_xlabel("Float Accuracy (%)", fontsize=12)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        # Single shared legend in the first panel.
        if not legend_ready:
            handles = [sc] + ([sc16] if _has_quant16(data) else [])
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=False)
            legend_ready = True

    # Add dataset indicators to the title
    kws_studies = [sd for sd in valid if "kws" in sd["name"].lower()]
    har_studies = [sd for sd in valid if "har" in sd["name"].lower()]
    if kws_studies and har_studies:
        title += f"\n(KWS: {len(kws_studies)} studies, HAR: {len(har_studies)} studies)"


    # Tighten layout
    plt.tight_layout()

    savefig(fig, title, "accuracy_grid")
