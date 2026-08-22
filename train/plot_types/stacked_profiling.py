from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .common import savefig


def create_stacked_profiling_plot(study_name, config_path, title, absolute=False):
    """
    Stacked bar chart of MCU operator latency across ALL MCU-tested trials.

    One bar per trial (sorted by total MCU latency ascending), with a segment
    per operator type. By default the y-axis is normalised to 100% so the
    operator *composition* is directly comparable across trials; with
    ``absolute=True`` the bars show actual summed latency per operator (ms).
    Total latency per trial is annotated above each bar.

    Parameters
    ----------
    study_name : str
        Optuna study name (used to locate results.json).
    config_path : str
        Path to the Hydra config YAML (for metadata; not otherwise used).
    title : str
        Used in plot title and saved file names.
    absolute : bool
        If True plot summed latency (ms) instead of normalised percentage.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "experiments" / study_name / "results.json"

    if not results_path.exists():
        print(f"  Error: No results.json found at {results_path}")
        return False

    with open(results_path) as f:
        results_data = json.load(f)

    trials = [d for d in results_data if d.get("mcu_profiling")]
    if not trials:
        print(f"  Error: No MCU profiling data in {results_path}")
        return False

    trials = sorted(trials, key=lambda d: sum(
        v["total_latency_ms"] for v in d["mcu_profiling"].values()))
    trial_nums = [d["trial_number"] for d in trials]
    totals = [sum(v["total_latency_ms"] for v in d["mcu_profiling"].values())
              for d in trials]

    # Union of operator types so colours stay consistent across trials.
    ops = sorted({op for d in trials for op in d["mcu_profiling"]})
    colors = cm.tab20(np.linspace(0, 1, len(ops)))
    op_color = dict(zip(ops, colors))

    low_ops = {d["trial_number"]: d["mcu_profiling"] for d in trials}

    n = len(trials)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n), 6))

    bottom = np.zeros(n)
    for op in ops:
        vals = np.array([low_ops[t][op]["total_latency_ms"] if op in low_ops[t] else 0.0
                         for t in trial_nums])
        if absolute:
            plot_vals = vals
        else:
            plot_vals = np.divide(vals, totals, out=np.zeros_like(vals),
                                  where=np.array(totals) > 0) * 100.0
        ax.bar(x, plot_vals, bottom=bottom, label=op, color=op_color[op],
               edgecolor="white", linewidth=0.4, zorder=3)
        bottom += plot_vals

    ylab = "Total Latency on MCU (ms)" if absolute else "Share of MCU Latency (%)"
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlabel("Trial (sorted by total latency)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    if absolute:
        ax.set_ylim(0, max(bottom) * 1.15)

    ax.set_xticks(x)
    ax.set_xticklabels(trial_nums, rotation=45, ha="right", fontsize=8)

    for xi, tot in zip(x, totals):
        ax.annotate(f"{tot:.1f}", (xi, bottom[xi]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    # Compact legend with percentages, sorted by mean share descending.
    means = {op: np.mean([low_ops[t][op]["total_latency_ms"] / tot
                          for t, tot in zip(trial_nums, totals)
                          if op in low_ops[t]]) * 100.0 for op in ops}
    order = sorted(ops, key=lambda op: -means[op])
    labels = [f"{op}  ({means[op]:.0f}%)" for op in order]
    ax.legend(labels, fontsize=7, title="Op type (mean share)",
              title_fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)

    fig.subplots_adjust(right=0.72)
    savefig(fig, title, "stacked_profiling" + ("_abs" if absolute else ""))
    return True
