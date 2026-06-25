"""
HPO Comparison: compare multiple Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for N multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py config/a.yaml config/b.yaml config/c.yaml config/d.yaml
  python plot_arch_search.py --plot pareto config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py --plot accuracy config/arch-mamba1-kws.yaml
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
os.makedirs(OUT_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(OUT_DIR, name)

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


def create_pareto_front_plot(studies_data, title):
    """
    Plot and save the Pareto front comparison figure for N studies.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'df', 'par' (Pareto-sorted DataFrame),
        'color' (base), 'color_par' (Pareto highlight), 'idx' (int).
    title : str
        Used in the plot title and saved file names.
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

    ax.set_xlabel("Latency on pc (us, lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)

    n_studies = len(studies_data)
    ax.set_title(f"{title}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    slug = slugify(title)
    fig1.savefig(fig_path(f"pareto_{slug}.png"), dpi=FIG_DPI)
    fig1.savefig(fig_path(f"pareto_{slug}.pdf"))
    print(f"Saved figures/pareto_{slug}.png and pareto_{slug}.pdf")


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
    fig.savefig(fig_path(f"accuracy_{slug}.pdf"))
    print(f"  Saved figures/accuracy_{slug}.png and .pdf")


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
        "--plot", "-p", choices=["pareto", "accuracy"], required=True,
        help="Which plot to create: 'pareto' (Pareto front comparison) or "
             "'accuracy' (float vs quantized accuracy per study)."
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

    elif args.plot == "pareto":
        # ── Pareto front plot ────────────────────────────────────────────────
        create_pareto_front_plot(studies_data, title)
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
