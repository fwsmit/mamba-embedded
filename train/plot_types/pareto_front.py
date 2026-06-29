import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.legend_handler import HandlerBase

from .common import savefig


ALPHA_ALL = 0.25
ALPHA_PARETO = 0.95
MARKER_ALL = "o"
MARKER_PAR = "D"


class _TwoMarkerProxy:
    """Proxy artist carrying the two colours for a combined legend entry."""
    def __init__(self, color_base, color_par):
        self.color_base = color_base
        self.color_par = color_par


class _TwoMarkerHandler(HandlerBase):
    """Legend handler that draws two markers side-by-side in one entry."""
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        cx1 = width * 0.25
        cx2 = width * 0.65
        cy  = height * 0.5

        return [
            Line2D([cx1], [cy], marker=MARKER_ALL, color="w",
                   markerfacecolor=orig_handle.color_base, alpha=ALPHA_ALL,
                   markersize=7, transform=trans),
            Line2D([cx2], [cy], marker=MARKER_PAR, color="w",
                   markerfacecolor=orig_handle.color_par, markeredgecolor="white",
                   markeredgewidth=0.6, markersize=9, transform=trans),
        ]

def create_mcu_pareto_plot(studies_data, title):
    """
    Plot PC Pareto front with MCU-tested models highlighted, plus a separate
    panel showing MCU accuracy vs MCU latency for those models.

    For each study, all PC trials are plotted faintly. Models that were selected
    for MCU testing (found in ``results.json``) are highlighted with a star
    marker. The companion panel plots those same models with their on-device
    accuracy and latency.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'df', 'par', 'results_data' (list of entries
        from results.json with ``mcu_accuracy`` and ``mcu_latency_ms``),
        'color', 'color_par', 'idx'.
    title : str
        Used in the plot title and saved file names.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax_pc = fig.add_subplot(gs[0])
    ax_mcu = fig.add_subplot(gs[1])

    legend_handles = []
    legend_labels = []

    # ── Pre-compute y-axis limits (before right panel so we can filter visible points) ──
    all_par_acc = np.concatenate([sd["par"]["accuracy"].values for sd in studies_data])
    if len(all_par_acc) > 0:
        lo_acc = all_par_acc.min()
        total_range = 1.0 - lo_acc
        pad_acc = total_range * 0.2 if total_range > 0 else 0.01
        bot_acc = max(0.0, lo_acc - pad_acc)
        top_acc = 1.0
    else:
        bot_acc = 0.0
        top_acc = 1.0

    for sd in studies_data:
        # ── Left panel: PC Pareto front ──────────────────────────────────────
        # All PC trials (faint)
        ax_pc.scatter(sd["df"]["latency"], sd["df"]["accuracy"],
                      color=sd["color"], alpha=ALPHA_ALL, s=22, marker=MARKER_ALL,
                      zorder=2)

        # Pareto-optimal points and step-line
        ax_pc.scatter(sd["par"]["latency"], sd["par"]["accuracy"],
                      color=sd["color_par"], alpha=ALPHA_PARETO, s=70,
                      marker=MARKER_PAR, edgecolors="white", linewidths=0.6,
                      zorder=4)
        ax_pc.step(sd["par"]["latency"], sd["par"]["accuracy"],
                   color=sd["color_par"], linewidth=1.8, where="post", zorder=3)

        # MCU-tested trials highlighted with star marker
        mcu_highlight_x = []
        mcu_highlight_y = []
        if sd.get("results_data"):
            for rd in sd["results_data"]:
                tn = rd["trial_number"]
                match = sd["df"][sd["df"]["number"] == tn]
                if len(match) > 0:
                    mcu_highlight_x.append(match.iloc[0]["latency"])
                    mcu_highlight_y.append(match.iloc[0]["accuracy"])

        if mcu_highlight_x:
            ax_pc.scatter(mcu_highlight_x, mcu_highlight_y,
                          color=sd["color_par"], alpha=1.0, s=130,
                          marker="*", edgecolors="red", linewidths=1.2,
                          zorder=5, label=f"{sd['name']} MCU-tested")
            handles_pc = [
                Line2D([0], [0], marker=MARKER_ALL, color="w",
                       markerfacecolor=sd["color"], alpha=ALPHA_ALL, markersize=7),
                Line2D([0], [0], marker=MARKER_PAR, color="w",
                       markerfacecolor=sd["color_par"], markeredgecolor="white",
                       markeredgewidth=0.6, markersize=9),
                Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=sd["color_par"], markeredgecolor="red",
                       markeredgewidth=1.2, markersize=11),
            ]
            legend_handles.extend(handles_pc)
            legend_labels.extend([f"{sd['name']} all", f"{sd['name']} Pareto", f"{sd['name']} MCU"])

        # Extend Pareto step lines to plot edges
        if len(sd["par"]) > 0:
            first_x = sd["par"]["latency"].iloc[0]
            first_y = sd["par"]["accuracy"].iloc[0]
            last_x  = sd["par"]["latency"].iloc[-1]
            last_y  = sd["par"]["accuracy"].iloc[-1]
            xlim = ax_pc.get_xlim()
            ylim = ax_pc.get_ylim()
            ax_pc.plot([last_x, xlim[1]], [last_y, last_y],
                       color=sd["color_par"], linewidth=1.8, zorder=3)
            ax_pc.plot([first_x, first_x], [first_y, ylim[0]],
                       color=sd["color_par"], linewidth=1.8, zorder=3)

        # ── Right panel: MCU accuracy vs latency ─────────────────────────────
        if sd.get("results_data"):
            mcu_pts = []
            for rd in sd["results_data"]:
                mcu_acc = rd.get("mcu_accuracy", np.nan)
                mcu_lat = rd.get("mcu_latency_ms", np.nan)
                float_acc = rd.get("float_accuracy", np.nan)
                if not np.isnan(mcu_acc) and not np.isnan(mcu_lat) and not np.isnan(float_acc):
                    mcu_pts.append((mcu_lat, mcu_acc, float_acc, int(rd["trial_number"])))

            if mcu_pts:
                mcu_pts.sort(key=lambda x: x[0])  # sort by latency
                # Only keep points where either float or MCU accuracy is visible in the plot
                visible_pts = [(lat, mcu_acc, float_acc, tn)
                               for lat, mcu_acc, float_acc, tn in mcu_pts
                               if (bot_acc <= float_acc / 100.0 <= top_acc)
                                  or (bot_acc <= mcu_acc / 100.0 <= top_acc)]

                if visible_pts:
                    lat_vals    = [p[0] for p in visible_pts]
                    mcu_acc_vals = [p[1] for p in visible_pts]

                    # Plot MCU accuracy as filled circles
                    ax_mcu.scatter(lat_vals, [a / 100.0 for a in mcu_acc_vals],
                                   color=sd["color_par"], alpha=0.9, s=70,
                                   marker="o", edgecolors="white", linewidths=0.6,
                                   zorder=4, label=f"{sd['name']} MCU")

    # ── Left panel decorations ───────────────────────────────────────────────
    ax_pc.set_ylim(bot_acc, top_acc)
    tick_start = np.ceil(bot_acc / 0.02) * 0.02
    ax_pc.set_yticks(np.arange(tick_start, 1.001, 0.02))
    ax_pc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    all_par_lat = np.concatenate([sd["par"]["latency"].values for sd in studies_data])
    if len(all_par_lat) > 0:
        lo, hi = all_par_lat.min(), all_par_lat.max()
        span = hi - lo
        pad_lat = span * 0.2 if span > 0 else (lo * 0.2 if lo > 0 else 10.0)
        ax_pc.set_xlim(lo - pad_lat, hi + pad_lat)

    ax_pc.set_xlabel("Latency on PC (\u00b5s, lower is better)", fontsize=11)
    ax_pc.set_ylabel("Accuracy (higher is better)", fontsize=11)
    ax_pc.set_title("PC Pareto Front (\u2605 = MCU-tested)", fontsize=12, fontweight="bold")
    ax_pc.grid(True, alpha=0.3, linestyle="--")
    if legend_handles:
        ax_pc.legend(handles=legend_handles, labels=legend_labels,
                     framealpha=0.9, fontsize=8)

    # ── Right panel decorations ──────────────────────────────────────────────
    ax_mcu.set_xlabel("Latency on MCU (\u00b5s, lower is better)", fontsize=11)
    ax_mcu.set_ylabel("Accuracy (higher is better)", fontsize=11)
    ax_mcu.set_title("MCU Accuracy vs Latency", fontsize=12, fontweight="bold")
    ax_mcu.grid(True, alpha=0.3, linestyle="--")

    # ── Right panel legend ─────────────────────────────────────────────────-
    # Collect unique legend entries across studies
    mcu_legend_handles = []
    mcu_legend_labels = []
    for sd in studies_data:
        if sd.get("results_data"):
            mcu_legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=sd["color_par"],
                       markeredgecolor="white", markeredgewidth=0.6, markersize=8),
            )
            mcu_legend_labels.append(
                f"{sd['name']} MCU Acc.",
            )
    if mcu_legend_handles:
        ax_mcu.legend(handles=mcu_legend_handles, labels=mcu_legend_labels,
                      framealpha=0.9, fontsize=8)

    # Use the same accuracy axis scaling as the PC panel (fraction [0,1] with PercentFormatter)
    # so the two panels are directly comparable
    pc_ylim = ax_pc.get_ylim()
    pc_yticks = ax_pc.get_yticks()
    ax_mcu.set_ylim(pc_ylim)
    ax_mcu.set_yticks(pc_yticks)
    ax_mcu.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    savefig(fig, title, "muc_pareto")


def create_pareto_front_plot(studies_data, title, use_mcu=False):
    """
    Plot and save the Pareto front comparison figure for N studies.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'df', 'par' (Pareto-sorted DataFrame),
        'color' (base), 'color_par' (Pareto highlight), 'idx' (int).
    title : str
        Used in the plot title and saved file names.
    use_mcu : bool
        If True, plot MCU accuracy vs MCU latency instead of PC objectives.
    """
    n_studies = len(studies_data)

    fig1, ax = plt.subplots(figsize=(9, 6))

    # ── All trials (faint) ────────────────────────────────────────────────────
    for sd in studies_data:
        ax.scatter(sd["df"]["latency"], sd["df"]["accuracy"],
                   color=sd["color"], alpha=ALPHA_ALL, s=22, marker=MARKER_ALL,
                   zorder=2)

    # ── Pareto-optimal points and step-lines ──────────────────────────────────
    for sd in studies_data:
        ax.scatter(sd["par"]["latency"], sd["par"]["accuracy"],
                   color=sd["color_par"], alpha=ALPHA_PARETO, s=70,
                   marker=MARKER_PAR, edgecolors="white", linewidths=0.6,
                   zorder=4, label=f"{sd['name']} Pareto")
        ax.step(sd["par"]["latency"], sd["par"]["accuracy"],
                color=sd["color_par"], linewidth=1.8, where="post", zorder=3)

    # ── Frame y-axis around the Pareto front accuracy, capped at 100% ──────
    all_par_acc = np.concatenate([sd["par"]["accuracy"].values for sd in studies_data])
    if len(all_par_acc) > 0:
        lo_acc = all_par_acc.min()
        total_range = 1.0 - lo_acc
        if total_range > 0:
            pad = total_range * 0.2
        else:
            pad = 0.01  # fallback when all have perfect accuracy
        bot = max(0.0, lo_acc - pad)
        ax.set_ylim(bot, 1.0)
        tick_start = np.ceil(bot / 0.02) * 0.02
        ax.set_yticks(np.arange(tick_start, 1.001, 0.02))
    else:
        ax.set_yticks(np.arange(0.80, 1.001, 0.02))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # ── Build legend — one entry per study, two markers side-by-side ─────────
    legend_elements = []
    for sd in studies_data:
        legend_elements.append(
            _TwoMarkerProxy(sd["color"], sd["color_par"])
        )

    ax.legend(handles=legend_elements,
              labels=[sd["name"] for sd in studies_data],
              handler_map={_TwoMarkerProxy: _TwoMarkerHandler()},
              framealpha=0.9, fontsize=9)

    # ── Frame x-axis around the Pareto front with padding ────────────────────
    all_par_lat = np.concatenate([sd["par"]["latency"].values for sd in studies_data])
    if len(all_par_lat) > 0:
        lo, hi = all_par_lat.min(), all_par_lat.max()
        span = hi - lo
        if span > 0:
            pad = span * 0.2
        else:
            pad = lo * 0.2 if lo > 0 else 10.0
        ax.set_xlim(lo - pad, hi + pad)

    # ── Extend each Pareto-front step line out to the right and bottom ────────
    for sd in studies_data:
        if len(sd["par"]) == 0:
            continue
        first_x = sd["par"]["latency"].iloc[0]   # leftmost (lowest-latency) Pareto point
        first_y = sd["par"]["accuracy"].iloc[0]  # highest accuracy
        last_x  = sd["par"]["latency"].iloc[-1]  # rightmost (highest-latency) Pareto point
        last_y  = sd["par"]["accuracy"].iloc[-1] # lowest accuracy
        x_right = ax.get_xlim()[1]
        y_bottom = ax.get_ylim()[0]
        # horizontal extension to the right edge of the plot (from the rightmost point)
        ax.plot([last_x, x_right], [last_y, last_y],
                color=sd["color_par"], linewidth=1.8, zorder=3)
        # vertical extension down to the bottom edge (from the leftmost point)
        ax.plot([first_x, first_x], [first_y, y_bottom],
                color=sd["color_par"], linewidth=1.8, zorder=3)

    xlabel = "Latency on MCU (ms, lower is better)" if use_mcu else "Latency on PC (ms, lower is better)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)

    n_studies = len(studies_data)
    ax.set_title(f"{title}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    savefig(fig1, title, "pareto_front")
