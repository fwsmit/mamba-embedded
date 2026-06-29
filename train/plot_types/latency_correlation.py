import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig


def create_latency_correlation_plot(studies_data, title):
    """
    Scatter plot of PC latency vs MCU latency for all MCU-tested trials.

    Each point represents a model that was profiled both on PC (via Optuna)
    and on the ESP32-S3 (via results.json).  A trend line (least-squares fit
    through the origin, i.e. MCU = k \u00b7 PC) is drawn as a dashed line,
    and the axes are adjusted to include (0, 0) so the constant factor can
    be visually assessed.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'study_name', 'df', 'par', 'results_data',
        'color', 'color_par', 'idx'.
    title : str
        Used in plot title and saved file names.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Collect all (pc_lat, mcu_lat, trial_num) across studies
    all_points = []
    legend_handles = []
    legend_labels = []

    for sd in studies_data:
        if not sd.get("results_data"):
            continue
        for rd in sd["results_data"]:
            tn = rd["trial_number"]
            mcu_lat = rd.get("mcu_latency_ms", np.nan)
            if np.isnan(mcu_lat):
                continue
            match = sd["df"][sd["df"]["number"] == tn]
            if len(match) == 0:
                continue
            pc_lat = match.iloc[0]["latency"]
            all_points.append((pc_lat, mcu_lat, tn, sd["color_par"], sd["name"]))

        if len([p for p in all_points if p[4] == sd["name"]]) > 0:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=sd["color_par"], markersize=8)
            )
            legend_labels.append(sd["name"])

    if not all_points:
        print("  No MCU-tested trials with both PC and MCU latency found.")
        ax.text(0.5, 0.5, "No MCU data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        fig.tight_layout()
        slug = slugify(title)
        fig.savefig(fig_path(f"latency_{slug}.png"), dpi=FIG_DPI)
        fig.savefig(fig_pdf_path(f"latency_{slug}.pdf"))
        print(f"Saved figures/latency_{slug}.png and figures/pdf/latency_{slug}.pdf")
        return

    pc_lats = [p[0] for p in all_points]
    mcu_lats = [p[1] for p in all_points]
    colors = [p[3] for p in all_points]

    ax.scatter(pc_lats, mcu_lats, c=colors, s=60, marker="o",
               edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate each point with its trial number
    for pc_lat, mcu_lat, tn, _, _ in all_points:
        ax.annotate(str(tn), (pc_lat, mcu_lat),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, alpha=0.8)

    # ── Trend line (ordinary least squares, y = m * x + b) ──────────────────
    x_arr = np.array(pc_lats, dtype=float)
    y_arr = np.array(mcu_lats, dtype=float)
    m, b = np.polyfit(x_arr, y_arr, 1)

    # ── Axis limits including origin ─────────────────────────────────────────
    def _auto_lim_origin(vals):
        lo, hi = 0.0, max(vals)
        span = hi - lo
        pad = span * 0.15 if span > 0 else (lo * 0.15 if lo > 0 else 10.0)
        return lo - pad * 0.3, hi + pad

    ax.set_xlim(_auto_lim_origin(pc_lats))
    ax.set_ylim(_auto_lim_origin(mcu_lats))

    ax.set_xlabel("Latency on PC (µs)", fontsize=11)
    ax.set_ylabel("Latency on MCU (ms)", fontsize=11)
    ax.set_title("PC Latency vs MCU Latency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Draw trend line across the full x-axis span
    x_full = np.linspace(*ax.get_xlim(), 200)
    ax.plot(x_full, m * x_full + b, color="gray", linewidth=1.5,
            linestyle="--", zorder=2, label=f"Trend: MCU = {m:.1f} \u00b7 PC + {b:.0f}")

    ax.legend(handles=legend_handles + [Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--")],
              labels=legend_labels + [f"Trend: MCU = {m:.1f} \u00b7 PC + {b:.0f}"],
              framealpha=0.9, fontsize=9)

    savefig(fig, title, "latency")
