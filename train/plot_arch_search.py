"""
HPO Comparison: mamba1-har vs mamba3-har
Pareto front and supporting analysis plots for two Optuna multi-objective studies.
Usage: python plot_hpo_comparison.py
"""

import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import numpy as np
import matplotlib.pyplot as plt
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
STUDY_M1     = "mamba1-har"
STUDY_M3     = "mamba3-har"

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

# ── Helper: shared param columns ──────────────────────────────────────────────
shared_params = sorted(set(df_m1.columns) & set(df_m3.columns) - {"accuracy", "latency", "number"})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Pareto Front Comparison  (main money plot)
# ═══════════════════════════════════════════════════════════════════════════════
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

ax.set_xlabel("Latency  (objective 2  — lower is better)", fontsize=11)
ax.set_ylabel("Accuracy  (objective 1  — higher is better)", fontsize=11)
ax.set_title("Pareto Front: Mamba-1 vs Mamba-3  (HAR)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
fig1.tight_layout()
fig1.savefig(fig_path("fig1_pareto_front.png"), dpi=FIG_DPI)
print("Saved figures/fig1_pareto_front.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Hypervolume Indicator over trial order
# ═══════════════════════════════════════════════════════════════════════════════
def compute_hypervolume_history(df, ref_acc, ref_lat):
    """
    Incremental 2-D hypervolume dominated by Pareto front up to each trial.
    ref point should be WORSE than all solutions (low acc, high lat).
    HV = area of objective space dominated relative to the reference point.
    We use a simple sweep-line for 2-D (max acc, min lat).
    """
    hvs = []
    best_acc = -np.inf
    best_lat =  np.inf
    front = []  # list of (lat, acc) on current front
    df_sorted = df.sort_values("number")

    for _, row in df_sorted.iterrows():
        acc, lat = row["accuracy"], row["latency"]
        front.append((lat, acc))
        # Rebuild non-dominated front
        front_arr = np.array(front)
        keep = []
        for i, (l, a) in enumerate(front_arr):
            dominated = any(
                front_arr[j, 1] >= a and front_arr[j, 0] <= l and
                (front_arr[j, 1] > a or front_arr[j, 0] < l)
                for j in range(len(front_arr)) if j != i
            )
            if not dominated:
                keep.append((l, a))
        front = keep
        # Compute hypervolume (2-D sweep)
        pts = sorted(keep, key=lambda x: x[0])   # sort by latency asc
        hv = 0.0
        prev_lat = ref_lat
        for lat_p, acc_p in pts:
            if acc_p > ref_acc and lat_p < ref_lat:
                hv += (prev_lat - lat_p) * (acc_p - ref_acc)
                prev_lat = lat_p
        hvs.append(hv)
    return df_sorted["number"].values, np.array(hvs)

# Reference point: worst possible (0 accuracy, max observed latency * 1.1)
ref_lat = max(df_m1["latency"].max(), df_m3["latency"].max()) * 1.1
ref_acc = 0.0

ns_m1, hv_m1 = compute_hypervolume_history(df_m1, ref_acc, ref_lat)
ns_m3, hv_m3 = compute_hypervolume_history(df_m3, ref_acc, ref_lat)

fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(ns_m1, hv_m1, color=PARETO_M1, linewidth=2, label=STUDY_M1)
ax.plot(ns_m3, hv_m3, color=PARETO_M3, linewidth=2, label=STUDY_M3)
ax.set_xlabel("Trial number", fontsize=11)
ax.set_ylabel("Hypervolume indicator", fontsize=11)
ax.set_title("Hypervolume Progress over HPO Search", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle="--")
fig2.tight_layout()
fig2.savefig(fig_path("fig2_hypervolume.png"), dpi=FIG_DPI)
print("Saved figures/fig2_hypervolume.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Accuracy & Latency distributions  (violin + box)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, metric, label, better in zip(
    axes,
    ["accuracy", "latency"],
    ["Accuracy", "Latency"],
    ["higher ↑", "lower ↓"],
):
    data = [df_m1[metric].values, df_m3[metric].values]
    colors = [COLOR_M1, COLOR_M3]
    labels = [STUDY_M1, STUDY_M3]
    positions = [1, 2]

    vp = ax.violinplot(data, positions=positions, showmedians=False,
                       showextrema=False, widths=0.6)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.35)

    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="grey"),
                    capprops=dict(color="grey"),
                    flierprops=dict(marker="x", color="grey", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    # Overlay Pareto-optimal points
    for df_s, mask, c, pos in [
        (df_m1, mask_m1, PARETO_M1, 1),
        (df_m3, mask_m3, PARETO_M3, 2),
    ]:
        ax.scatter(
            np.full(mask.sum(), pos),
            df_s[metric][mask],
            color=c, s=30, zorder=5, marker=MARKER_PAR,
            edgecolors="white", linewidths=0.5,
            label="Pareto pts" if pos == 1 else None,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"{label} distribution  ({better})", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

fig3.suptitle("Objective Distributions: Mamba-1 vs Mamba-3", fontsize=13, fontweight="bold", y=1.01)
fig3.tight_layout()
fig3.savefig(fig_path("fig3_distributions.png"), dpi=FIG_DPI, bbox_inches="tight")
print("Saved figures/fig3_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Parallel Coordinates of Pareto-front hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
if shared_params:
    all_pareto = pd.concat([
        par_m1.assign(study=STUDY_M1),
        par_m3.assign(study=STUDY_M3),
    ], ignore_index=True)

    cols_to_plot = shared_params + ["accuracy", "latency"]

    # Encode each column to [0,1]: numeric → min-max; categorical → integer code
    normed = all_pareto[cols_to_plot].copy()
    tick_labels_per_col = {}   # for annotating categorical axes

    for col in cols_to_plot:
        if pd.api.types.is_numeric_dtype(normed[col]):
            lo, hi = normed[col].min(), normed[col].max()
            normed[col] = (normed[col] - lo) / (hi - lo + 1e-12)
            tick_labels_per_col[col] = None   # numeric: no special tick labels
        else:
            categories = sorted(normed[col].dropna().unique().tolist())
            cat_map = {c: i / max(len(categories) - 1, 1) for i, c in enumerate(categories)}
            normed[col] = normed[col].map(cat_map)
            tick_labels_per_col[col] = categories  # strings to annotate on axis

    fig4, ax = plt.subplots(figsize=(max(10, len(cols_to_plot) * 1.6), 5))
    ax.set_xlim(-0.5, len(cols_to_plot) - 0.5)
    ax.set_ylim(-0.05, 1.05)

    color_map = {STUDY_M1: PARETO_M1, STUDY_M3: PARETO_M3}
    for _, row in all_pareto.iterrows():
        c = color_map[row["study"]]
        ys = [normed.loc[row.name, col] for col in cols_to_plot]
        ax.plot(range(len(cols_to_plot)), ys, color=c, alpha=0.55, linewidth=1.5)

    # Annotate categorical axes with their actual string values
    for x_idx, col in enumerate(cols_to_plot):
        cats = tick_labels_per_col.get(col)
        if cats is not None:
            for i, cat_name in enumerate(cats):
                y_pos = i / max(len(cats) - 1, 1)
                ax.text(x_idx + 0.04, y_pos, str(cat_name),
                        fontsize=6, va="center", color="dimgrey")

    ax.set_xticks(range(len(cols_to_plot)))
    ax.set_xticklabels(cols_to_plot, rotation=25, ha="right", fontsize=9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["min", "25%", "50%", "75%", "max"], fontsize=8)
    ax.grid(True, axis="x", alpha=0.4, linestyle="--")

    for x in range(len(cols_to_plot)):
        ax.axvline(x, color="grey", linewidth=0.5, alpha=0.4)

    legend_elements = [
        Line2D([0], [0], color=PARETO_M1, linewidth=2, label=STUDY_M1),
        Line2D([0], [0], color=PARETO_M3, linewidth=2, label=STUDY_M3),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.set_title("Parallel Coordinates — Pareto-Optimal Hyperparameters (normalised)",
                 fontsize=12, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(fig_path("fig4_parallel_coords.png"), dpi=FIG_DPI)
    print("Saved figures/fig4_parallel_coords.png")
else:
    print("No shared hyperparameters found — skipping parallel coordinates plot.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Per-parameter scatter vs accuracy (Pareto points coloured by latency)
# ═══════════════════════════════════════════════════════════════════════════════
numeric_shared = [p for p in shared_params
                  if pd.api.types.is_numeric_dtype(df_m1[p])
                  and pd.api.types.is_numeric_dtype(df_m3[p])]

if numeric_shared:
    shared_params_fig5 = numeric_shared
    n_params = len(shared_params_fig5)
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    fig5, axes5 = plt.subplots(nrows, ncols * 2,
                                figsize=(ncols * 2 * 3.5, nrows * 3.2),
                                squeeze=False)

    lat_all = pd.concat([df_m1["latency"], df_m3["latency"]])
    norm_lat = Normalize(vmin=lat_all.min(), vmax=lat_all.max())
    cmap = plt.cm.plasma_r

    for idx, param in enumerate(shared_params_fig5):
        row_i = idx // ncols
        col_base = (idx % ncols) * 2   # two axes per param (one per study)

        for study_idx, (df_s, mask, study_name) in enumerate([
            (df_m1, mask_m1, STUDY_M1),
            (df_m3, mask_m3, STUDY_M3),
        ]):
            ax = axes5[row_i][col_base + study_idx]

            if param not in df_s.columns:
                ax.set_visible(False)
                continue

            sc = ax.scatter(
                df_s[param], df_s["accuracy"],
                c=df_s["latency"], cmap=cmap, norm=norm_lat,
                alpha=0.4, s=18, zorder=2
            )
            # Highlight Pareto
            ax.scatter(
                df_s[param][mask], df_s["accuracy"][mask],
                c=df_s["latency"][mask], cmap=cmap, norm=norm_lat,
                s=55, marker=MARKER_PAR, edgecolors="white",
                linewidths=0.6, zorder=4, alpha=0.95
            )
            ax.set_xlabel(param, fontsize=8)
            ax.set_ylabel("Accuracy" if study_idx == 0 else "", fontsize=8)
            ax.set_title(study_name, fontsize=8, color=PARETO_M1 if study_idx == 0 else PARETO_M3,
                         fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, linestyle="--")

    # Hide unused axes
    for idx in range(len(shared_params_fig5), ncols * nrows):
        row_i = idx // ncols
        col_base = (idx % ncols) * 2
        for k in range(2):
            axes5[row_i][col_base + k].set_visible(False)

    # Shared colorbar — add a slim dedicated axes to the right of the figure
    # so it never overlaps any subplot
    fig5.subplots_adjust(right=0.88)
    cbar_ax = fig5.add_axes([0.90, 0.15, 0.015, 0.7])   # [left, bottom, width, height]
    sm = ScalarMappable(cmap=cmap, norm=norm_lat)
    sm.set_array([])
    cbar = fig5.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Latency (colour)", fontsize=9)

    fig5.suptitle("Hyperparameter vs Accuracy (coloured by Latency) — Pareto points as ◆",
                  fontsize=12, fontweight="bold")
    fig5.savefig(fig_path("fig5_param_scatter.png"), dpi=FIG_DPI, bbox_inches="tight")
    print("Saved figures/fig5_param_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Best-trial summary table  (text figure)
# ═══════════════════════════════════════════════════════════════════════════════
def best_pareto_summary(df, mask, study_name):
    p = df[mask].copy()
    best_acc = p.loc[p["accuracy"].idxmax()]
    best_lat = p.loc[p["latency"].idxmin()]
    # "knee" – closest to utopia point (1, 0) in normalised space
    p_norm_acc = (p["accuracy"] - p["accuracy"].min()) / (p["accuracy"].max() - p["accuracy"].min() + 1e-9)
    p_norm_lat = (p["latency"]  - p["latency"].min())  / (p["latency"].max()  - p["latency"].min()  + 1e-9)
    dist = np.sqrt((1 - p_norm_acc)**2 + p_norm_lat**2)
    knee = p.iloc[dist.values.argmin()]
    return {
        "study":         study_name,
        "n_trials":      len(df),
        "n_pareto":      mask.sum(),
        "best_acc":      f"{best_acc['accuracy']:.4f}  (lat={best_acc['latency']:.4f})",
        "best_lat":      f"{best_lat['latency']:.4f}  (acc={best_lat['accuracy']:.4f})",
        "knee_acc":      f"{knee['accuracy']:.4f}",
        "knee_lat":      f"{knee['latency']:.4f}",
    }

sum_m1 = best_pareto_summary(df_m1, mask_m1, STUDY_M1)
sum_m3 = best_pareto_summary(df_m3, mask_m3, STUDY_M3)

fig6, ax6 = plt.subplots(figsize=(10, 3))
ax6.axis("off")

rows_data = []
col_keys = ["n_trials", "n_pareto", "best_acc", "best_lat", "knee_acc", "knee_lat"]
col_labels = ["Trials", "Pareto pts", "Best Accuracy (latency)", "Best Latency (accuracy)",
              "Knee Accuracy", "Knee Latency"]

for s in [sum_m1, sum_m3]:
    rows_data.append([s[k] for k in col_keys])

table = ax6.table(
    cellText=rows_data,
    rowLabels=[sum_m1["study"], sum_m3["study"]],
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Colour the row headers
for row_idx, color in enumerate([PARETO_M1, PARETO_M3]):
    table[(row_idx + 1, -1)].set_facecolor(color)
    table[(row_idx + 1, -1)].set_text_props(color="white", fontweight="bold")

for col_idx in range(len(col_labels)):
    table[(0, col_idx)].set_facecolor("#2D2D2D")
    table[(0, col_idx)].set_text_props(color="white", fontweight="bold")

ax6.set_title("Summary: Pareto Front Statistics", fontsize=12, fontweight="bold", pad=14)
fig6.tight_layout()
fig6.savefig(fig_path("fig6_summary_table.png"), dpi=FIG_DPI, bbox_inches="tight")
print("Saved figures/fig6_summary_table.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 – Hyperparameter Importance  (per objective, both studies)
# ═══════════════════════════════════════════════════════════════════════════════
OBJECTIVE_NAMES = ["Accuracy", "Latency"]   # human-readable names for values[0], values[1]
OBJ_COLORS      = ["#5B8DD9", "#E8834C"]    # blue=accuracy, orange=latency

def get_importances(study, obj_idx):
    """Return {param: importance} for a given objective index."""
    try:
        imp = optuna.importance.get_param_importances(
            study,
            target=lambda t: t.values[obj_idx],
        )
        return dict(imp)
    except Exception as e:
        print(f"  Warning: importance failed for obj {obj_idx}: {e}")
        return {}

fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, study, study_name, bar_colors in zip(
    axes7,
    [study_m1, study_m3],
    [STUDY_M1, STUDY_M3],
    [[PARETO_M1, PARETO_M3], [COLOR_M1, COLOR_M3]],   # use distinct shades per study
):
    imp_per_obj = [get_importances(study, i) for i in range(len(OBJECTIVE_NAMES))]

    # Union of all params, sorted by mean importance descending
    all_params = sorted(
        set().union(*[d.keys() for d in imp_per_obj]),
        key=lambda p: np.mean([d.get(p, 0) for d in imp_per_obj]),
    )

    if not all_params:
        ax.text(0.5, 0.5, "Not enough trials\nfor importance estimate",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(study_name, fontsize=11, fontweight="bold")
        continue

    n_params   = len(all_params)
    n_obj      = len(OBJECTIVE_NAMES)
    bar_height = 0.35
    y_positions = np.arange(n_params)

    for obj_idx, (obj_name, imp_dict) in enumerate(zip(OBJECTIVE_NAMES, imp_per_obj)):
        values = [imp_dict.get(p, 0.0) for p in all_params]
        offset = (obj_idx - (n_obj - 1) / 2) * bar_height
        bars = ax.barh(
            y_positions + offset, values,
            height=bar_height,
            color=OBJ_COLORS[obj_idx],
            alpha=0.85,
            label=obj_name,
        )
        # Value labels
        for bar, v in zip(bars, values):
            if v > 0.005:
                ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", va="center", fontsize=7.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_params, fontsize=9)
    ax.set_xlabel("Hyperparameter Importance", fontsize=10)
    ax.set_title(study_name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, title="Objective", title_fontsize=8)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)

fig7.suptitle("Hyperparameter Importance by Objective", fontsize=13, fontweight="bold")
fig7.tight_layout()
fig7.savefig(fig_path("fig7_hp_importance.png"), dpi=FIG_DPI, bbox_inches="tight")
print("Saved figures/fig7_hp_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 – Terminator Improvement  (expected improvement + regret bound)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_terminator_series(study):
    """
    Compute per-trial terminator improvement and regret-bound error using
    optuna.terminator if available, otherwise fall back to hypervolume delta.
    Returns (trial_numbers, improvements, errors).
    """
    try:
        from optuna.terminator import (
            RegretBoundEvaluator,
            TerminatorImprovementEvaluator,
        )
        imp_eval = TerminatorImprovementEvaluator()
        err_eval = RegretBoundEvaluator()

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial_nums, improvements, errors = [], [], []

        for i in range(2, len(completed) + 1):   # need at least 2 trials
            subset = completed[:i]
            try:
                imp = imp_eval.evaluate(trials=subset, study_direction=None)
                err = err_eval.evaluate(trials=subset, study_direction=None)
            except Exception:
                try:
                    # Older API: evaluate(study)
                    imp = imp_eval.evaluate(study)
                    err = err_eval.evaluate(study)
                except Exception:
                    continue
            trial_nums.append(subset[-1].number)
            improvements.append(float(imp))
            errors.append(float(err))

        if trial_nums:
            return np.array(trial_nums), np.array(improvements), np.array(errors)
    except ImportError:
        pass

    # ── Fallback: hypervolume delta ──────────────────────────────────────────
    df  = trials_df(study)
    ref_lat = df["latency"].max() * 1.1
    ns, hvs = compute_hypervolume_history(df, 0.0, ref_lat)
    delta    = np.abs(np.diff(hvs, prepend=hvs[0]))
    # Smooth error estimate: rolling std of delta
    window   = max(3, len(delta) // 8)
    errors   = pd.Series(delta).rolling(window, min_periods=1).std().fillna(0).values
    return ns, delta, errors

fig8, axes8 = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

for ax, study, study_name, imp_color, err_color in zip(
    axes8,
    [study_m1, study_m3],
    [STUDY_M1, STUDY_M3],
    [PARETO_M1, PARETO_M3],
    [COLOR_M1, COLOR_M3],
):
    trial_ns, improvements, errors = compute_terminator_series(study)

    if len(trial_ns) == 0:
        ax.text(0.5, 0.5, "Not enough trials", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
    else:
        ax.plot(trial_ns, improvements, color=imp_color, linewidth=2,
                marker="o", markersize=4, label="Terminator Improvement")
        ax.plot(trial_ns, errors, color=err_color, linewidth=2,
                marker="o", markersize=4, label="Regret Bound / Error")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_xlabel("Trial", fontsize=10)
    ax.set_ylabel("Terminator Improvement", fontsize=10)
    ax.set_title(study_name, fontsize=11, fontweight="bold")

fig8.suptitle("Terminator Improvement Plot  (Accuracy ↑ · Latency ↓)",
              fontsize=13, fontweight="bold")
fig8.tight_layout()
fig8.savefig(fig_path("fig8_terminator.png"), dpi=FIG_DPI, bbox_inches="tight")
print("Saved figures/fig8_terminator.png")

print("\nAll figures saved. Done.")
plt.show()
