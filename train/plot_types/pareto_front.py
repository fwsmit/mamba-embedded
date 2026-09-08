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

ACCURACY_LABEL = "Accuracy on validation set"
LATENCY_MCU_LABEL = "Latency on MCU (ms, lower is better)"
LATENCY_PC_LABEL = "Latency on PC ($\\mu$s, lower is better)"
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12


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

def create_mcu_pareto_plot(studies_data, title,
                           accuracy_field="test_quantized_accuracy",
                           accuracy_label=None):
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
        from results.json,
        'color', 'color_par', 'idx'.
    title : str
        Used in the plot title and saved file names.
    accuracy_field : str
        results.json field plotted as the y-axis accuracy in the right panel
        (e.g. ``test_quantized_accuracy``, ``test_quantized_accuracy_int16``,
        ``test_quantized_accuracy_strat``, ``test_float_accuracy``).
    accuracy_label : str, optional
        Y-axis label for the right panel.  Falls back to
        ``Accuracy on validation set`` when not given.
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
        bot_acc = max(0.86, lo_acc - pad_acc)
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

        # ── Determine MCU-tested trials with valid MCU data ────────────────
        mcu_pts = []
        if sd.get("results_data"):
            for rd in sd["results_data"]:
                mcu_acc = rd.get(accuracy_field, np.nan)
                mcu_lat = rd.get("mcu_latency_ms", np.nan)
                float_acc = rd.get("test_float_accuracy", np.nan)
                if not np.isnan(mcu_acc) and not np.isnan(mcu_lat) and not np.isnan(float_acc):
                    mcu_pts.append((mcu_lat, mcu_acc, float_acc, int(rd["trial_number"])))

            mcu_pts.sort(key=lambda x: x[0])
        # Trials that actually appear on the right panel
        mcu_tns = {p[3] for p in mcu_pts}

        # ── Left panel: MCU-tested trials highlighted with star marker ───────
        mcu_highlight_x = []
        mcu_highlight_y = []
        mcu_highlight_tn = []
        if sd.get("results_data"):
            for rd in sd["results_data"]:
                tn = rd["trial_number"]
                if tn not in mcu_tns:
                    continue
                match = sd["df"][sd["df"]["number"] == tn]
                if len(match) > 0:
                    mcu_highlight_x.append(match.iloc[0]["latency"])
                    mcu_highlight_y.append(match.iloc[0]["accuracy"])
                    mcu_highlight_tn.append(tn)

        if mcu_highlight_x:
            ax_pc.scatter(mcu_highlight_x, mcu_highlight_y,
                          color=sd["color_par"], alpha=1.0, s=130,
                          marker="*", edgecolors="red", linewidths=1.2,
                          zorder=5, label=f"{sd['name']} MCU-tested")
            for lx, ly, tn in zip(mcu_highlight_x, mcu_highlight_y, mcu_highlight_tn):
                ax_pc.annotate(str(tn), (lx, ly),
                               textcoords="offset points", xytext=(8, 8),
                               fontsize=10, fontweight="bold", color="red",
                               zorder=6)
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
        if mcu_pts:
            lat_vals    = [p[0] for p in mcu_pts]
            mcu_acc_vals = [p[1] for p in mcu_pts]

            # Plot MCU accuracy as filled circles
            ax_mcu.scatter(lat_vals, [a / 100.0 for a in mcu_acc_vals],
                           color=sd["color_par"], alpha=0.9, s=70,
                           marker="o", edgecolors="white", linewidths=0.6,
                           zorder=4, label=f"{sd['name']} MCU")
            for lat, acc, _, tn_pt in mcu_pts:
                ax_mcu.annotate(str(tn_pt), (lat, acc / 100.0),
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=10, fontweight="bold",
                                color=sd["color_par"], zorder=5)

    # ── Left panel decorations ───────────────────────────────────────────────
    ax_pc.set_ylim(bot_acc, top_acc)
    tick_start = np.ceil(bot_acc / 0.02) * 0.02
    ax_pc.set_yticks(np.arange(tick_start, 1.001, 0.02))
    ax_pc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    all_par_lat = np.concatenate([sd["par"]["latency"].values for sd in studies_data])
    if len(all_par_lat) > 0:
        hi = all_par_lat.max()
        span = hi
        pad_lat = span * 0.2 if span > 0 else 10.0
        ax_pc.set_xlim(0, hi + pad_lat)

    ax_pc.set_xlabel(LATENCY_PC_LABEL, fontsize=11)
    ax_pc.set_ylabel(ACCURACY_LABEL, fontsize=11)
    ax_pc.grid(True, alpha=0.3, linestyle="--")
    if legend_handles:
        ax_pc.legend(handles=legend_handles, labels=legend_labels,
                     framealpha=0.9, fontsize=8)

    # ── Right panel decorations ──────────────────────────────────────────────
    ax_mcu.set_xlabel(LATENCY_MCU_LABEL, fontsize=11)
    ax_mcu.set_ylabel(accuracy_label or ACCURACY_LABEL, fontsize=11)
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


def create_pareto_front_plot(studies_data, title, use_mcu=False, top_acc=1.0, ylim=None, xlim=None):
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
    top_acc : float or None
        Upper y-axis limit as a fraction (default 1.0 = 100%%). Pass None to
        frame the axis around the actual accuracy range of the trials instead
        of capping at 100%%.
    ylim : tuple of float or None
        Optional fixed y-axis range in percent (e.g. (84, 90)); overrides the
        automatic axis framing computed above.
    xlim : tuple of float or None
        Optional fixed x-axis range in latency units (e.g. (0, 80) for PC µs);
        overrides the automatic axis framing computed below.
    """
    n_studies = len(studies_data)

    # Render all text with LaTeX so the figure matches the fonts of the LaTeX
    # thesis it will be embedded in. helvet is loaded with the same scaling as
    # the thesis (\usepackage[scaled=.92]{helvet}) so sans-serif text uses
    # Helvetica instead of Computer Modern Sans.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage[scaled=.92]{helvet}"

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

    # ── Frame y-axis around the Pareto front accuracy ───────────────────────
    all_par_acc = np.concatenate([sd["par"]["accuracy"].values for sd in studies_data])
    if len(all_par_acc) > 0:
        lo_acc = all_par_acc.min()
        if top_acc is None:
            # auto: frame around the Pareto front points themselves
            hi_acc = all_par_acc.max()
            spread = hi_acc - lo_acc
            pad = spread * 0.1 if spread > 0 else 0.01
            bot = np.floor(max(0.0, lo_acc - pad) / 0.02) * 0.02
            top = np.ceil(min(1.0, hi_acc + pad) / 0.02) * 0.02
        else:
            spread = top_acc - lo_acc
            pad = spread * 0.2 if spread > 0 else 0.01
            bot = max(0.86, lo_acc - pad)
            top = top_acc
        ax.set_ylim(bot, top)
    else:
        pass
    if ylim is not None:
        ax.set_ylim(ylim[0] / 100.0, ylim[1] / 100.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # ── Build legend — one entry per study, two markers side-by-side ─────────
    legend_elements = []
    for sd in studies_data:
        legend_elements.append(
            _TwoMarkerProxy(sd["color"], sd["color_par"])
        )

    ax.legend(handles=legend_elements,
              labels=[sd["name"] for sd in studies_data],
              handler_map={_TwoMarkerProxy: _TwoMarkerHandler()},
              loc="lower right",
              framealpha=0.9, fontsize=13)

    # ── Frame x-axis starting at latency 0, with padding on the right ─────────
    all_par_lat = np.concatenate([sd["par"]["latency"].values for sd in studies_data])
    if len(all_par_lat) > 0:
        hi = all_par_lat.max()
        span = hi
        pad = span * 0.2 if span > 0 else 10.0
        ax.set_xlim(0, hi + pad)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

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

    xlabel = LATENCY_MCU_LABEL if use_mcu else LATENCY_PC_LABEL
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ACCURACY_LABEL, fontsize=AXIS_LABEL_SIZE)

    ax.tick_params(axis='both', which='major', labelsize=13)

    n_studies = len(studies_data)
    ax.grid(True, alpha=0.3, linestyle="--")

    savefig(fig1, title, "pareto_front")
