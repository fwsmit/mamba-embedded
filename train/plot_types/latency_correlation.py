import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker

from .common import savefig


def _dataset_label(title):
    """Extract dataset label(s) from a study-name-style title string."""
    parts = []
    if "har" in title.lower().split("-"):
        parts.append("HAR")
    if "kws" in title.lower().split("-"):
        parts.append("KWS")
    return " & ".join(parts) if parts else None


def create_latency_correlation_plot(studies_data, title, use_param_size=False):
    """
    Scatter plot of PC latency vs MCU latency for all MCU-tested trials.

    Each point represents a model that was profiled both on PC (via Optuna)
    and on the ESP32-S3 (via results.json).  A trend line (least-squares fit
    through the origin, i.e. MCU = k \u00b7 PC) is drawn as a dashed line,
    and the axes are adjusted to include (0, 0) so the constant factor can
    be visually assessed.

    When *use_param_size* is True, the x-axis shows parameter size (bytes)
    instead of PC latency.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'study_name', 'df', 'par', 'results_data',
        'color', 'color_par', 'idx'.
    title : str
        Used in plot title and saved file names.
    use_param_size : bool, optional
        If True, correlate MCU latency with parameter size instead of PC latency.
        Default is False.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Collect all (x_val, mcu_lat, trial_num) across studies
    all_points = []
    legend_handles = []
    legend_labels = []

    if use_param_size:
        x_label = "Parameter Size (bytes)"
        trend_label_fmt = "MCU = {m:.3f} \u00b7 param + {b:.0f}"
    else:
        x_label = "Latency on PC (\u00b5s)"
        trend_label_fmt = "MCU = {m:.1f} \u00b7 PC + {b:.0f}"

    for sd in studies_data:
        if not sd.get("results_data"):
            continue
        for rd in sd["results_data"]:
            tn = rd["trial_number"]
            mcu_lat = rd.get("mcu_latency_ms", np.nan)
            if np.isnan(mcu_lat):
                continue
            if use_param_size:
                x_val = rd.get("param_size_bytes", np.nan)
                if np.isnan(x_val):
                    continue
            else:
                match = sd["df"][sd["df"]["number"] == tn]
                if len(match) == 0:
                    continue
                x_val = match.iloc[0]["latency"]
            all_points.append((x_val, mcu_lat, tn, sd["color_par"], sd["name"]))

        if len([p for p in all_points if p[4] == sd["name"]]) > 0:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=sd["color_par"], markersize=8)
            )
            legend_labels.append(sd["name"])

    if not all_points:
        print("  No MCU-tested trials with both PC and MCU latency found.")
        exit(1)

    x_vals = [p[0] for p in all_points]
    mcu_lats = [p[1] for p in all_points]
    colors = [p[3] for p in all_points]

    ax.scatter(x_vals, mcu_lats, c=colors, s=60, marker="o",
               edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate each point with its trial number
    for x_val, mcu_lat, tn, _, _ in all_points:
        ax.annotate(str(tn), (x_val, mcu_lat),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, alpha=0.8)

    # ── Trend line (ordinary least squares, y = m * x + b) ──────────────────
    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(mcu_lats, dtype=float)
    m, b = np.polyfit(x_arr, y_arr, 1)

    # ── Axis limits including origin ─────────────────────────────────────────
    def _auto_lim_origin(vals):
        lo, hi = 0.0, max(vals)
        span = hi - lo
        pad = span * 0.15 if span > 0 else (lo * 0.15 if lo > 0 else 10.0)
        return lo - pad * 0.3, hi + pad

    ax.set_xlim(_auto_lim_origin(x_vals))
    ax.set_ylim(_auto_lim_origin(mcu_lats))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Latency on MCU (ms)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Draw trend line across the full x-axis span
    x_full = np.linspace(*ax.get_xlim(), 200)
    trend_label = trend_label_fmt.format(m=m, b=b)
    ax.plot(x_full, m * x_full + b, color="gray", linewidth=1.5,
            linestyle="--", zorder=2, label=trend_label)

    ax.legend(handles=legend_handles + [Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--")],
              labels=legend_labels + [trend_label],
              framealpha=0.9, fontsize=9)
    
    if use_param_size:
        filename = "latency_param"
    else:
        filename = "latency"

    savefig(fig, title, filename)
