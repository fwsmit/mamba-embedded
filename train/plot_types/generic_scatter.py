"""
Generic scatter plot: plot any numeric field from results.json against another.

Usage (via plot_arch_search.py):
  python -m train.plot_arch_search --plot scatter \\
      --x-field param_size_bytes --y-field mcu_latency_ms \\
      --x-label "Parameter Size (bytes)" --y-label "Latency on MCU (ms)" \\
      config/har/*
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig, slugify, fig_path, fig_pdf_path, FIG_DPI


def create_generic_scatter_plot(studies_data, title, x_field, y_field,
                                x_label="", y_label=""):
    """
    Scatter plot of *x_field* vs *y_field* extracted from each study's
    ``results.json`` data.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'study_name', 'df', 'par', 'results_data',
        'color', 'color_par', 'idx'.
    title : str
        Used in the plot title and saved file names.
    x_field : str
        Field name in ``results_data`` entries for the x-axis (e.g. ``param_size_bytes``).
    y_field : str
        Field name in ``results_data`` entries for the y-axis (e.g. ``mcu_latency_ms``).
    x_label : str
        Label for the x-axis.  If empty, falls back to the field name.
    y_label : str
        Label for the y-axis.  If empty, falls back to the field name.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    all_points = []
    legend_handles = []
    legend_labels = []

    for sd in studies_data:
        if not sd.get("results_data"):
            continue
        for rd in sd["results_data"]:
            x_val = rd.get(x_field, np.nan)
            y_val = rd.get(y_field, np.nan)
            tn = rd.get("trial_number", -1)
            if not np.isnan(x_val) and not np.isnan(y_val):
                all_points.append((x_val, y_val, tn, sd["color_par"], sd["name"]))

        if len([p for p in all_points if p[4] == sd["name"]]) > 0:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=sd["color_par"], markersize=8)
            )
            legend_labels.append(sd["name"])

    if not all_points:
        print(f"  No data found for fields '{x_field}' and '{y_field}' across all studies.")
        exit(1)

    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    colors = [p[3] for p in all_points]

    ax.scatter(x_vals, y_vals, c=colors, s=60, marker="o",
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlabel(x_label or x_field, fontsize=11)
    ax.set_ylabel(y_label or y_field, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Format x-axis with K/M suffix for byte-sized values
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if abs(x) >= 1000 else f"{x:.0f}"
    ))

    ax.legend(handles=legend_handles,
              labels=legend_labels,
              framealpha=0.9, fontsize=9)

    savefig(fig, title, "scatter")
