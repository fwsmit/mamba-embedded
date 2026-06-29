from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from .common import savefig


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
    repo_root = Path(__file__).resolve().parent.parent.parent
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
    ops = sorted(profiling.items(), key=lambda x: x[1]["total_latency_ms"], reverse=True)
    op_names = [o[0] for o in ops]
    latencies = [o[1]["total_latency_ms"] for o in ops]
    counts = [o[1]["count"] for o in ops]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(op_names))
    width = 0.65

    bars = ax.bar(x, latencies, width, color="#4C9BE8", edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(op_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total Latency on MCU (ms)", fontsize=11)
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

    savefig(fig, title, "profiling")
    return True
