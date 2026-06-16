"""
HPO Comparison: compare multiple Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for N multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py config/a.yaml config/b.yaml config/c.yaml config/d.yaml
"""

import argparse
import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
import os
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
def create_pareto_front_plot(studies_data):
    """
    Plot and save the Pareto front comparison figure for N studies.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'df', 'par' (Pareto-sorted DataFrame),
        'color' (base), 'color_par' (Pareto highlight), 'idx' (int).
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

    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # ── Build legend ──────────────────────────────────────────────────────────
    legend_elements = []
    for sd in studies_data:
        legend_elements.append(
            Line2D([0], [0], marker=MARKER_PAR, color="w",
                   markerfacecolor=sd["color_par"], markersize=9,
                   label=f"{sd['name']} — Pareto front")
        )
        legend_elements.append(
            Line2D([0], [0], marker=MARKER_ALL, color="w",
                   markerfacecolor=sd["color"], markersize=7, alpha=0.7,
                   label=f"{sd['name']} — all trials")
        )

    ax.legend(handles=legend_elements, framealpha=0.9, fontsize=9)

    ax.set_xlabel("Latency on pc (us, lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)

    n_studies = len(studies_data)
    title_suffix = " vs ".join(sd["name"] for sd in studies_data)
    ax.set_title(f"HPO Pareto front comparison: {title_suffix}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(fig_path("fig1_pareto_front.png"), dpi=FIG_DPI)
    fig1.savefig(fig_path("fig1_pareto_front.pdf"))
    print("Saved figures/fig1_pareto_front.png and fig1_pareto_front.pdf")


def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding Optuna study name."""
    cfg = OmegaConf.load(config_path)
    return f"{cfg.MODEL}-{cfg.DATASET}-{cfg.EXPERIMENT_NAME}" if cfg.get("EXPERIMENT_NAME") else f"{cfg.MODEL}-{cfg.DATASET}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Optuna studies from their Hydra config files"
    )
    parser.add_argument(
        "configs", nargs="+",
        help="Paths to Hydra config YAML files (at least 1, up to any number)"
    )
    args = parser.parse_args()

    n = len(args.configs)
    if n == 0:
        parser.error("At least one config path is required.")

    if n > len(COLORS):
        print(f"Warning: {n} studies provided but only {len(COLORS)} colours defined. "
              f"Colours will be recycled from the beginning.")

    # ── Gather study data ─────────────────────────────────────────────────────
    studies_data = []
    for i, config_path in enumerate(args.configs):
        name = study_name_from_config(config_path)
        color_base, color_par = COLORS[i % len(COLORS)]
        print(f"Study {i+1}: {name}  (from {config_path})  → colour {color_base}")

        print("  Loading study …")
        study = load_study(name)
        df = trials_df(study)
        mask = pareto_mask(df)

        # Sort Pareto front by latency for clean step-line drawing
        par = df[mask].copy().sort_values("latency")

        print(f"  {name}: {len(df)} trials, {len(par)} Pareto-optimal")

        studies_data.append({
            "name": name,
            "df": df,
            "par": par,
            "color": color_base,
            "color_par": color_par,
            "idx": i,
        })

    # ── Plot ──────────────────────────────────────────────────────────────────
    create_pareto_front_plot(studies_data)

    print("\nAll figures saved. Done.")
    plt.show()


if __name__ == "__main__":
    main()