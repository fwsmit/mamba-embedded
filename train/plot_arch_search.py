"""
HPO Comparison: compare multiple Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for N multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py config/a.yaml config/b.yaml config/c.yaml config/d.yaml
  python plot_arch_search.py --plot pareto config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py --plot accuracy config/arch-mamba1-kws.yaml
  python plot_arch_search.py --plot pareto --use-mcu config/arch-mamba1-har.yaml
  python plot_arch_search.py --plot latency config/arch-mamba1-har.yaml
"""

import argparse
import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
import os
import json
import re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures"
PDF_DIR = os.path.join(OUT_DIR, "pdf")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(OUT_DIR, name)

def fig_pdf_path(name):
    return os.path.join(PDF_DIR, name)

# ── Config (defaults, overridden by command-line args) ───────────────────────
DB_URL       = "sqlite:///mamba_hpo.db"

# Colour palette for up to N studies (base, pareto)
COLORS = [
    ("#4C9BE8", "#1A5FA8"),   # blue
    ("#E8834C", "#A84F1A"),   # orange
    ("#4CAF50", "#2E7D32"),   # green
    ("#9C27B0", "#6A1B9A"),   # purple
    ("#009688", "#00695C"),   # teal
    ("#E91E63", "#AD1457"),   # pink
]

ALPHA_ALL    = 0.25
ALPHA_PARETO = 0.95
MARKER_ALL   = "o"
MARKER_PAR   = "D"
FIG_DPI      = 150


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


# ── Load studies ──────────────────────────────────────────────────────────────
def load_study(name):
    return optuna.load_study(study_name=name, storage=DB_URL)

def trials_df(study):
    """Return a DataFrame of completed trials with objectives and params."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(t.params)
        row["accuracy"] = t.values[0]   # objective 0
        row["latency"]  = t.values[1]   # objective 1
        row["number"]   = t.number
        rows.append(row)
    return pd.DataFrame(rows)

def pareto_mask(df):
    """Boolean mask for Pareto-optimal trials (max accuracy, min latency)."""
    acc = df["accuracy"].values
    lat = df["latency"].values
    n   = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good on both and strictly better on one
            if acc[j] >= acc[i] and lat[j] <= lat[i] and (acc[j] > acc[i] or lat[j] < lat[i]):
                dominated[i] = True
                break
    return ~dominated



# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Pareto Front Comparison  (main money plot)
# ═══════════════════════════════════════════════════════════════════════════════
def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = re.sub(r"[ _]+", "-", s)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – MCU Pareto Comparison  (PC Pareto front + MCU-tested highlights)
# ═══════════════════════════════════════════════════════════════════════════════
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
        from results.json with ``mcu_accuracy`` and ``mcu_latency_us``),
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
                mcu_lat = rd.get("mcu_latency_us", np.nan)
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
                    float_acc_vals = [p[2] for p in visible_pts]

                    # Draw a vertical line from float accuracy to MCU accuracy for each trial
                    for lat, float_acc, mcu_acc in zip(lat_vals, float_acc_vals, mcu_acc_vals):
                        ax_mcu.plot([lat, lat], [float_acc / 100.0, mcu_acc / 100.0],
                                    color=sd["color_par"], alpha=0.4, linewidth=1.2,
                                    linestyle="-", zorder=2)

                    # Plot float accuracy as fainter crosses at the same MCU latency
                    ax_mcu.scatter(lat_vals, [a / 100.0 for a in float_acc_vals],
                                   color=sd["color_par"], alpha=0.35, s=50,
                                   marker="x", linewidths=1.2,
                                   zorder=3, label=f"{sd['name']} float")
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
            mcu_legend_handles.extend([
                Line2D([0], [0], marker="x", color="w", markerfacecolor=sd["color_par"],
                       markeredgecolor=sd["color_par"], alpha=0.35, markersize=7),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=sd["color_par"],
                       markeredgecolor="white", markeredgewidth=0.6, markersize=8),
            ])
            mcu_legend_labels.extend([
                f"{sd['name']} Float Acc.",
                f"{sd['name']} MCU Acc.",
            ])
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

    fig.tight_layout()
    slug = slugify(title)
    fig.savefig(fig_path(f"mcu_pareto_{slug}.png"), dpi=FIG_DPI)
    fig.savefig(fig_pdf_path(f"mcu_pareto_{slug}.pdf"))
    print(f"Saved figures/mcu_pareto_{slug}.png and figures/pdf/mcu_pareto_{slug}.pdf")


def create_profiling_plot(study_name, config_path, trial_number, title):
    """
    Vertical bar chart of MCU operator profiling for a specific trial.

    Plots each operator type's total latency on the MCU, sorted descending,
    with a second bar showing the count (scaled to max latency for visual
    comparison). Annotations display exact values.

    Parameters
    ----------
    study_name : str
        Optuna study name (used to locate results.json).
    config_path : str
        Path to the Hydra config YAML (for metadata).
    trial_number : int
        Trial number to plot profiling for.
    title : str
        Used in plot title and saved file names.
    """
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "experiments" / study_name / "results.json"

    if not results_path.exists():
        print(f"  Error: No results.json found at {results_path}")
        return False

    with open(results_path) as f:
        results_data = json.load(f)

    # Find the trial entry
    trial_entry = None
    for d in results_data:
        if d["trial_number"] == trial_number:
            trial_entry = d
            break

    if trial_entry is None:
        print(f"  Error: Trial {trial_number} not found in {results_path}")
        return False

    profiling = trial_entry.get("mcu_profiling")
    if not profiling:
        print(f"  Error: No mcu_profiling data for trial {trial_number}")
        return False

    # Sort operators by total latency descending
    ops = sorted(profiling.items(), key=lambda x: x[1]["total_latency_us"], reverse=True)
    op_names = [o[0] for o in ops]
    latencies = [o[1]["total_latency_us"] for o in ops]
    counts = [o[1]["count"] for o in ops]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(op_names))
    width = 0.65

    bars = ax.bar(x, latencies, width, color="#4C9BE8", edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(op_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total Latency on MCU (µs)", fontsize=11)
    ax.set_xlabel("Operator Type", fontsize=11)
    ax.set_title(f"MCU Operator Profiling — Trial {trial_number}  ({title})",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add a bit of headroom
    max_lat = max(latencies) if latencies else 1
    ax.set_ylim(0, max_lat * 1.15)

    # Also show total latency and average latency in a text box
    total = sum(latencies)
    avg_per_op = total / len(latencies)
    info_text = f"Total: {total} µs  |  Avg/op: {avg_per_op:.0f} µs"
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor="lightyellow",
                                              edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    slug = slugify(f"{title}_trial{trial_number}")
    fig.savefig(fig_path(f"profiling_{slug}.png"), dpi=FIG_DPI)
    fig.savefig(fig_pdf_path(f"profiling_{slug}.pdf"))
    print(f"Saved figures/profiling_{slug}.png and figures/pdf/profiling_{slug}.pdf")
    return True


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
            mcu_lat = rd.get("mcu_latency_us", np.nan)
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
    ax.set_ylabel("Latency on MCU (µs)", fontsize=11)
    ax.set_title("PC Latency vs MCU Latency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Draw trend line across the full x-axis span
    x_full = np.linspace(*ax.get_xlim(), 200)
    ax.plot(x_full, m * x_full + b, color="gray", linewidth=1.5,
            linestyle="--", zorder=2, label=f"Trend: MCU = {m:.1f} \u00b7 PC + {b:.0f}")

    ax.legend(handles=legend_handles + [Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--")],
              labels=legend_labels + [f"Trend: MCU = {m:.1f} \u00b7 PC + {b:.0f}"],
              framealpha=0.9, fontsize=9)

    fig.tight_layout()
    slug = slugify(title)
    fig.savefig(fig_path(f"latency_{slug}.png"), dpi=FIG_DPI)
    fig.savefig(fig_pdf_path(f"latency_{slug}.pdf"))
    print(f"Saved figures/latency_{slug}.png and figures/pdf/latency_{slug}.pdf")


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

    xlabel = "Latency on MCU (\u00b5s, lower is better)" if use_mcu else "Latency on PC (\u00b5s, lower is better)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)

    n_studies = len(studies_data)
    ax.set_title(f"{title}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    slug = slugify(title)
    suffix = "_mcu" if use_mcu else ""
    fig1.savefig(fig_path(f"pareto_{slug}{suffix}.png"), dpi=FIG_DPI)
    fig1.savefig(fig_pdf_path(f"pareto_{slug}{suffix}.pdf"))
    print(f"Saved figures/pareto_{slug}{suffix}.png and figures/pdf/pareto_{slug}{suffix}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Float vs Quantized Accuracy Comparison
# ═══════════════════════════════════════════════════════════════════════════════
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

    fig.tight_layout()
    slug = slugify(title)
    fig.savefig(fig_path(f"accuracy_{slug}.png"), dpi=FIG_DPI)
    fig.savefig(fig_pdf_path(f"accuracy_{slug}.pdf"))
    print(f"  Saved figures/accuracy_{slug}.png and figures/pdf/accuracy_{slug}.pdf")


def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding Optuna study name."""
    cfg = OmegaConf.load(config_path)
    return f"{cfg.MODEL}-{cfg.DATASET}-{cfg.EXPERIMENT_NAME}" if cfg.get("EXPERIMENT_NAME") else f"{cfg.MODEL}-{cfg.DATASET}"


def load_study_meta(config_path: str) -> dict:
    """Load a Hydra config YAML and return metadata: study_name, display_name.

    The display name is set from the optional ``plot_description`` field in the
    config file (useful for giving configs clearer labels in Pareto front plots
    when comparing multiple experiments). Falls back to the auto-generated study
    name if ``plot_description`` is not present.
    """
    cfg = OmegaConf.load(config_path)
    study_name = study_name_from_config(config_path)
    display_name = cfg.get("plot_description") or study_name
    return {"study_name": study_name, "display_name": display_name}


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Optuna studies from their Hydra config files"
    )
    parser.add_argument(
        "configs", nargs="+",
        help="Paths to Hydra config YAML files (at least 1, up to any number)"
    )
    parser.add_argument(
        "--plot", "-p", choices=["pareto", "accuracy", "mcu_pareto", "latency", "profiling"], required=True,
        help="Which plot to create: 'pareto' (Pareto front comparison), "
             "'accuracy' (float vs quantized accuracy per study), "
             "'mcu_pareto' (PC Pareto front with MCU-tested highlights + MCU perf plot), "
             "'latency' (PC latency vs MCU latency scatter plot), or "
             "'profiling' (MCU operator profiling bar chart for a specific trial)."
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Title of the plot and base name for saved files. "
             "If not provided, derived from the study names."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the plot(s) on screen (default: only save to disk)."
    )
    parser.add_argument(
        "--mcu", action="store_true",
        help="Include on-device MCU accuracy bars in the accuracy plot (default: off)."
    )
    parser.add_argument(
        "--trial", type=int, default=None,
        help="Trial number for the 'profiling' plot."
    )
    parser.add_argument(
        "--use-mcu", action="store_true",
        help="Use MCU accuracy and latency from results.json instead of PC objectives "
             "in the Pareto front plot (only valid with --plot pareto)."
    )
    args = parser.parse_args()

    n = len(args.configs)
    if n == 0:
        parser.error("Provide at least one config path.")

    if n > len(COLORS):
        print(f"Warning: {n} studies provided but only {len(COLORS)} colours defined. "
              f"Colours will be recycled from the beginning.")

    repo_root = Path(__file__).resolve().parent.parent

    # ── Derive title if not provided ─────────────────────────────────────────
    title = args.title or " vs ".join(
        study_name_from_config(cp) for cp in args.configs
    )

    # ── Gather study data ─────────────────────────────────────────────────────
    studies_data = []
    for i, config_path in enumerate(args.configs):
        meta = load_study_meta(config_path)
        name = meta["study_name"]
        display_name = meta["display_name"]
        color_base, color_par = COLORS[i % len(COLORS)]
        print(f"Study {i+1}: {name}  (from {config_path})  → colour {color_base}")
        if display_name != name:
            print(f"  → Plot label: {display_name}")

        print("  Loading study …")
        study = load_study(name)
        df = trials_df(study)
        mask = pareto_mask(df)

        # Sort Pareto front by latency for clean step-line drawing
        par = df[mask].copy().sort_values("latency")

        print(f"  {name}: {len(df)} trials, {len(par)} Pareto-optimal")

        studies_data.append({
            "name": display_name,
            "study_name": name,
            "df": df,
            "par": par,
            "color": color_base,
            "color_par": color_par,
            "idx": i,
        })

    # ── Create the requested plot ───────────────────────────────────────────
    plot_created = False

    if args.plot == "accuracy":
        # ── Accuracy comparison (inferred from config) ────────────────────────
        for sd in studies_data:
            name = sd["name"]
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading accuracy results from {results_path} …")
                with open(results_path) as f:
                    results_data = json.load(f)
                create_accuracy_comparison_plot(name, results_data, title, show_mcu=args.mcu)
                plot_created = True
            else:
                print(f"  No results.json found at {results_path}, skipping.")

    elif args.plot == "mcu_pareto":
        # ── MCU Pareto plot (PC front + MCU performance) ────────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_mcu_pareto_plot(studies_data, title)
        plot_created = True

    elif args.plot == "latency":
        # ── PC latency vs MCU latency scatter plot ─────────────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_latency_correlation_plot(studies_data, title)
        plot_created = True

    elif args.plot == "pareto":
        # ── Pareto front plot (optionally with MCU data) ────────────────────
        if args.use_mcu:
            print("  Using MCU data from results.json instead of PC objectives ...")
            for sd in studies_data:
                results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
                if not results_path.exists():
                    print(f"  Warning: No results.json at {results_path}, cannot use MCU data for {sd['name']}.")
                    continue
                with open(results_path) as f:
                    results_data = json.load(f)
                mcu_rows = []
                for rd in results_data:
                    mcu_acc = rd.get("mcu_accuracy", np.nan)
                    mcu_lat = rd.get("mcu_latency_us", np.nan)
                    if not np.isnan(mcu_acc) and not np.isnan(mcu_lat):
                        mcu_rows.append({
                            "accuracy": mcu_acc / 100.0,  # % → fraction
                            "latency": mcu_lat,
                            "number": rd["trial_number"],
                        })
                if not mcu_rows:
                    print(f"  Warning: No valid MCU results for {sd['name']}, keeping PC data.")
                    continue
                mcu_df = pd.DataFrame(mcu_rows)
                mask = pareto_mask(mcu_df)
                par = mcu_df[mask].copy().sort_values("latency")
                print(f"  {sd['name']}: {len(mcu_df)} MCU-tested trials, {len(par)} Pareto-optimal")
                sd["df"] = mcu_df
                sd["par"] = par

        create_pareto_front_plot(studies_data, title, use_mcu=args.use_mcu)
        plot_created = True

    elif args.plot == "profiling":
        # ── MCU operator profiling bar chart for a specific trial ──────────
        if args.trial is None:
            parser.error("--trial is required when using --plot profiling")
        if len(args.configs) != 1:
            parser.error("profiling plot requires exactly one config file.")

        meta = load_study_meta(args.configs[0])
        study_name = meta["study_name"]
        display_name = meta["display_name"]
        print(f"  Profiling trial {args.trial} from study {study_name}")

        if create_profiling_plot(study_name, args.configs[0], args.trial, display_name):
            plot_created = True

    if not plot_created:
        print("\nNo plots created.")
    else:
        print("\nAll requested figures saved.")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
