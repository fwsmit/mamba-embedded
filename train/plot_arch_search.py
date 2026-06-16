"""
HPO Comparison: compare two Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for two Optuna multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
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
STUDY_A      = "mamba-1-kws-1"
STUDY_B      = "mamba-1-kws-2"

COLOR_A      = "#4C9BE8"   # blue  – first study
COLOR_B      = "#E8834C"   # orange – second study
PARETO_A     = "#1A5FA8"
PARETO_B     = "#A84F1A"
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
def create_pareto_front_plot(df_a, df_b, par_a, par_b):
    """Plot and save the Pareto front comparison figure."""
    fig1, ax = plt.subplots(figsize=(9, 6))

    # All trials (faint)
    ax.scatter(df_a["latency"], df_a["accuracy"],
               color=COLOR_A, alpha=ALPHA_ALL, s=22, marker=MARKER_ALL, zorder=2)
    ax.scatter(df_b["latency"], df_b["accuracy"],
               color=COLOR_B, alpha=ALPHA_ALL, s=22, marker=MARKER_ALL, zorder=2)

    # Pareto-optimal points
    ax.scatter(par_a["latency"], par_a["accuracy"],
               color=PARETO_A, alpha=ALPHA_PARETO, s=70, marker=MARKER_PAR,
               edgecolors="white", linewidths=0.6, zorder=4, label=f"{STUDY_A} Pareto")
    ax.scatter(par_b["latency"], par_b["accuracy"],
               color=PARETO_B, alpha=ALPHA_PARETO, s=70, marker=MARKER_PAR,
               edgecolors="white", linewidths=0.6, zorder=4, label=f"{STUDY_B} Pareto")

    # Pareto step-lines (staircase front)
    ax.step(par_a["latency"], par_a["accuracy"],
            color=PARETO_A, linewidth=1.8, where="post", zorder=3)
    ax.step(par_b["latency"], par_b["accuracy"],
            color=PARETO_B, linewidth=1.8, where="post", zorder=3)

    # ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Legend with all-trials proxy
    legend_elements = [
        Line2D([0], [0], marker=MARKER_PAR, color="w", markerfacecolor=PARETO_A,
               markersize=9, label=f"{STUDY_A} — Pareto front"),
        Line2D([0], [0], marker=MARKER_PAR, color="w", markerfacecolor=PARETO_B,
               markersize=9, label=f"{STUDY_B} — Pareto front"),
        Line2D([0], [0], marker=MARKER_ALL, color="w", markerfacecolor=COLOR_A,
               markersize=7, alpha=0.7, label=f"{STUDY_A} — all trials"),
        Line2D([0], [0], marker=MARKER_ALL, color="w", markerfacecolor=COLOR_B,
               markersize=7, alpha=0.7, label=f"{STUDY_B} — all trials"),
    ]
    ax.legend(handles=legend_elements, framealpha=0.9, fontsize=9)

    ax.set_xlabel("Latency on pc (us, lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)
    ax.set_title("Hyperparameter optimization: first vs second study", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(fig_path("fig1_pareto_front.png"), dpi=FIG_DPI)
    print("Saved figures/fig1_pareto_front.png")


def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding Optuna study name."""
    cfg = OmegaConf.load(config_path)
    return f"{cfg.MODEL}-{cfg.DATASET}-{cfg.EXPERIMENT_NAME}" if cfg.get("EXPERIMENT_NAME") else f"{cfg.MODEL}-{cfg.DATASET}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Optuna studies from their Hydra config files"
    )
    parser.add_argument("config_a", help="Path to the first Hydra config YAML")
    parser.add_argument("config_b", help="Path to the second Hydra config YAML")
    args = parser.parse_args()

    global STUDY_A, STUDY_B
    STUDY_A = study_name_from_config(args.config_a)
    STUDY_B = study_name_from_config(args.config_b)
    print(f"Study A: {STUDY_A}  (from {args.config_a})")
    print(f"Study B: {STUDY_B}  (from {args.config_b})")

    # ── Load ──────────────────────────────────────────────────────────────────────
    print("Loading studies …")
    study_a = load_study(STUDY_A)
    study_b = load_study(STUDY_B)

    df_a = trials_df(study_a)
    df_b = trials_df(study_b)

    mask_a = pareto_mask(df_a)
    mask_b = pareto_mask(df_b)

    print(f"  {STUDY_A}: {len(df_a)} trials, {mask_a.sum()} Pareto-optimal")
    print(f"  {STUDY_B}: {len(df_b)} trials, {mask_b.sum()} Pareto-optimal")

    # Sort Pareto fronts by latency for clean line drawing
    def pareto_sorted(df, mask):
        p = df[mask].copy()
        return p.sort_values("latency")

    par_a = pareto_sorted(df_a, mask_a)
    par_b = pareto_sorted(df_b, mask_b)

    create_pareto_front_plot(df_a, df_b, par_a, par_b)

    print("\nAll figures saved. Done.")
    plt.show()


if __name__ == "__main__":
    main()
