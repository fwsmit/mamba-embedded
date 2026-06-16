"""
HPO Comparison: mamba1-har vs mamba3-har
Pareto front and supporting analysis plots for two Optuna multi-objective studies.
Usage: python plot_hpo_comparison.py
"""

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
import os
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(OUT_DIR, name)

# ── Config ────────────────────────────────────────────────────────────────────
DB_URL       = "sqlite:///mamba_hpo.db"
STUDY_M1     = "mamba-1-kws"
STUDY_M3     = "mamba-1-kws-2"

COLOR_M1     = "#4C9BE8"   # blue  – Mamba-1
COLOR_M3     = "#E8834C"   # orange – Mamba-3
PARETO_M1    = "#1A5FA8"
PARETO_M3    = "#A84F1A"
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
def create_pareto_front_plot(df_m1, df_m3, par_m1, par_m3):
    """Plot and save the Pareto front comparison figure."""
    fig1, ax = plt.subplots(figsize=(9, 6))

    # All trials (faint)
    ax.scatter(df_m1["latency"], df_m1["accuracy"],
               color=COLOR_M1, alpha=ALPHA_ALL, s=22, marker=MARKER_ALL, zorder=2)
    ax.scatter(df_m3["latency"], df_m3["accuracy"],
               color=COLOR_M3, alpha=ALPHA_ALL, s=22, marker=MARKER_ALL, zorder=2)

    # Pareto-optimal points
    ax.scatter(par_m1["latency"], par_m1["accuracy"],
               color=PARETO_M1, alpha=ALPHA_PARETO, s=70, marker=MARKER_PAR,
               edgecolors="white", linewidths=0.6, zorder=4, label=f"{STUDY_M1} Pareto")
    ax.scatter(par_m3["latency"], par_m3["accuracy"],
               color=PARETO_M3, alpha=ALPHA_PARETO, s=70, marker=MARKER_PAR,
               edgecolors="white", linewidths=0.6, zorder=4, label=f"{STUDY_M3} Pareto")

    # Pareto step-lines (staircase front)
    ax.step(par_m1["latency"], par_m1["accuracy"],
            color=PARETO_M1, linewidth=1.8, where="post", zorder=3)
    ax.step(par_m3["latency"], par_m3["accuracy"],
            color=PARETO_M3, linewidth=1.8, where="post", zorder=3)

    # ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Legend with all-trials proxy
    legend_elements = [
        Line2D([0], [0], marker=MARKER_PAR, color="w", markerfacecolor=PARETO_M1,
               markersize=9, label=f"{STUDY_M1} — Pareto front"),
        Line2D([0], [0], marker=MARKER_PAR, color="w", markerfacecolor=PARETO_M3,
               markersize=9, label=f"{STUDY_M3} — Pareto front"),
        Line2D([0], [0], marker=MARKER_ALL, color="w", markerfacecolor=COLOR_M1,
               markersize=7, alpha=0.7, label=f"{STUDY_M1} — all trials"),
        Line2D([0], [0], marker=MARKER_ALL, color="w", markerfacecolor=COLOR_M3,
               markersize=7, alpha=0.7, label=f"{STUDY_M3} — all trials"),
    ]
    ax.legend(handles=legend_elements, framealpha=0.9, fontsize=9)

    ax.set_xlabel("Latency on pc (us, lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy  (higher is better)", fontsize=11)
    ax.set_title("Hyperparameter optimization: Mamba-1 vs Mamba-1 (improved dataset (KWS)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(fig_path("fig1_pareto_front.png"), dpi=FIG_DPI)
    print("Saved figures/fig1_pareto_front.png")


def main():
    # ── Load ──────────────────────────────────────────────────────────────────────
    print("Loading studies …")
    study_m1 = load_study(STUDY_M1)
    study_m3 = load_study(STUDY_M3)

    df_m1 = trials_df(study_m1)
    df_m3 = trials_df(study_m3)

    mask_m1 = pareto_mask(df_m1)
    mask_m3 = pareto_mask(df_m3)

    print(f"  {STUDY_M1}: {len(df_m1)} trials, {mask_m1.sum()} Pareto-optimal")
    print(f"  {STUDY_M3}: {len(df_m3)} trials, {mask_m3.sum()} Pareto-optimal")

    # Sort Pareto fronts by latency for clean line drawing
    def pareto_sorted(df, mask):
        p = df[mask].copy()
        return p.sort_values("latency")

    par_m1 = pareto_sorted(df_m1, mask_m1)
    par_m3 = pareto_sorted(df_m3, mask_m3)

    create_pareto_front_plot(df_m1, df_m3, par_m1, par_m3)

    print("\nAll figures saved. Done.")
    plt.show()


if __name__ == "__main__":
    main()
