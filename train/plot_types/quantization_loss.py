import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .common import savefig


def create_quantization_loss_plot(studies_data, title):
    """
    Compare quantization loss (float_accuracy - quantized_accuracy) across multiple studies.

    Generates a two-panel figure:
    - Left panel: Grouped bar chart showing mean quantization loss per study with
      individual trial points overlaid as a strip plot.
    - Right panel: Float vs quantized accuracy scatter plot (one point per trial)
      annotated with the diagonal (no loss) and a trend line per study.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'study_name', 'results_data' (list of entries
        from results.json with ``float_accuracy`` and ``quantized_accuracy``),
        'color', 'color_par', 'idx'.
    title : str
        Used in the plot title and saved file names.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    ax_loss = fig.add_subplot(gs[0])
    ax_scatter = fig.add_subplot(gs[1])

    # ── Collect per-study losses ────────────────────────────────────────────
    study_losses = []      # list of lists of loss values per study
    study_means = []       # mean loss per study
    study_stds = []        # std loss per study
    valid_studies = []     # studies with data

    for sd in studies_data:
        if not sd.get("results_data"):
            continue
        losses = []
        for rd in sd["results_data"]:
            fa = rd.get("float_accuracy", np.nan)
            qa = rd.get("quantized_accuracy", np.nan)
            if not np.isnan(fa) and not np.isnan(qa):
                losses.append(fa - qa)
        if not losses:
            continue
        study_losses.append(losses)
        study_means.append(np.mean(losses))
        study_stds.append(np.std(losses))
        valid_studies.append(sd)

    if not valid_studies:
        print("  No quantization loss data found across any study.")
        exit(1)

    n_studies = len(valid_studies)
    x_pos = np.arange(n_studies)

    # ── Left panel: Bar chart of mean loss + individual points ────────────
    colors_bar = [sd["color_par"] for sd in valid_studies]
    bars = ax_loss.bar(x_pos, study_means, width=0.55,
                       color=colors_bar, edgecolor="white", linewidth=0.8,
                       alpha=0.85, zorder=3)

    # Error bars (std)
    ax_loss.errorbar(x_pos, study_means, yerr=study_stds,
                     fmt="none", ecolor="gray", capsize=4, capthick=1.2,
                     elinewidth=1.2, zorder=4)

    # Strip plot: individual trial losses overlaid
    for i, losses in enumerate(study_losses):
        # Jitter for visibility
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(losses))
        ax_loss.scatter(x_pos[i] + jitter, losses,
                        color=valid_studies[i]["color"], alpha=0.6, s=30,
                        marker="o", edgecolors="white", linewidths=0.3,
                        zorder=5)

    ax_loss.set_xticks(x_pos)
    ax_loss.set_xticklabels([sd["name"] for sd in valid_studies],
                            rotation=30, ha="right", fontsize=9)
    ax_loss.set_ylabel("Quantization Loss (float \u2212 quantized, %)", fontsize=11)
    ax_loss.set_title("Quantization Loss per Study", fontsize=12, fontweight="bold")
    ax_loss.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate bars with mean value
    for bar, mean_val in zip(bars, study_means):
        ax_loss.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{mean_val:.1f}%", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")

    # Extend y-axis a bit for annotation headroom
    all_losses = [l for subl in study_losses for l in subl]
    max_loss = max(all_losses) if all_losses else 1.0
    ax_loss.set_ylim(0, max_loss * 1.4)

    # ── Right panel: Float vs quantized accuracy scatter ─────────────────
    for sd in valid_studies:
        xs = []
        ys = []
        for rd in sd["results_data"]:
            fa = rd.get("float_accuracy", np.nan)
            qa = rd.get("quantized_accuracy", np.nan)
            if not np.isnan(fa) and not np.isnan(qa):
                xs.append(fa)
                ys.append(qa)
        if not xs:
            continue

        ax_scatter.scatter(xs, ys, color=sd["color_par"], alpha=0.7, s=40,
                           marker="o", edgecolors="white", linewidths=0.4,
                           zorder=3, label=sd["name"])

        # Trend line (linear fit, y = m*x + b)
        x_arr = np.array(xs)
        y_arr = np.array(ys)
        m, b = np.polyfit(x_arr, y_arr, 1)
        x_fit = np.linspace(min(xs), max(xs), 100)
        ax_scatter.plot(x_fit, m * x_fit + b, color=sd["color_par"],
                        linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)

    # Diagonal (no loss)
    diag_min = min(
        min(rd.get("float_accuracy", 100) for sd in valid_studies for rd in sd["results_data"]),
        min(rd.get("quantized_accuracy", 0) for sd in valid_studies for rd in sd["results_data"])
    )
    diag_max = 100.0
    ax_scatter.plot([diag_min, diag_max], [diag_min, diag_max],
                    color="gray", linewidth=1.0, linestyle=":",
                    alpha=0.5, zorder=1, label="No loss (y=x)")

    ax_scatter.set_xlabel("Float Accuracy (%)", fontsize=11)
    ax_scatter.set_ylabel("Quantized Accuracy (%)", fontsize=11)
    ax_scatter.set_title("Float vs Quantized Accuracy", fontsize=12, fontweight="bold")
    ax_scatter.grid(True, alpha=0.3, linestyle="--")
    ax_scatter.legend(fontsize=8, framealpha=0.9, loc="lower right")

    # Equal aspect so the diagonal is at 45 degrees
    ax_scatter.set_aspect("equal", adjustable="datalim")
    ax_scatter.set_xlim(diag_min - 2, diag_max + 2)
    ax_scatter.set_ylim(diag_min - 2, diag_max + 2)

    savefig(fig, title, "quant_loss")
