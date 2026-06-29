import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig

def create_param_vs_accuracy_plot(studies_data, title):
    fig, ax = plt.subplots(figsize=(9, 7))

    all_points = []
    legend_handles = []
    legend_labels = []

    for sd in studies_data:
        if not sd.get("results_data"):
            continue
        for rd in sd["results_data"]:
            param_size = rd.get("param_size_bytes", np.nan)
            mcu_lat = rd.get("mcu_accuracy", np.nan)
            tn = rd.get("trial_number", -1)
            if not np.isnan(param_size) and not np.isnan(mcu_lat):
                all_points.append((param_size, mcu_lat, tn, sd["color_par"], sd["name"]))

        if len([p for p in all_points if p[4] == sd["name"]]) > 0:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=sd["color_par"], markersize=8)
            )
            legend_labels.append(sd["name"])

    if not all_points:
        print("  No MCU-tested trials with both param_size and MCU accuracy found.")
        exit(1)

    param_sizes = [p[0] for p in all_points]
    mcu_lats = [p[1] for p in all_points]
    colors = [p[3] for p in all_points]

    ax.scatter(param_sizes, mcu_lats, c=colors, s=60, marker="o",
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlabel("Parameter Size (bytes, int8 quantized)", fontsize=11)
    ax.set_ylabel("Accuracy on MCU (%)", fontsize=11)
    ax.set_title(f"{title}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Format x-axis with K suffix for readability
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    ax.legend(handles=legend_handles + [Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--")],
              labels=legend_labels,
              framealpha=0.9, fontsize=9)

    savefig(fig, title, "param_accuracy")
