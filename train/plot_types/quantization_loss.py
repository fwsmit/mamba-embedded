import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .common import savefig, beeswarm_box, LINTHRESH

# Strategies: (display label, result-field key, colour, marker). The colour
# mapping (blue/purple/green) is kept consistent across all plots; distinct
# markers add a grayscale / colour-blind-safe secondary channel.
STRATEGIES = [
    ("8-bit (prc)",    "test_quantized_accuracy",       "#4C9BE8", "o"),
    ("8-bit (kl-tqt)", "test_quantized_accuracy_strat", "#9C27B0", "s"),
    ("16-bit (prc)",   "test_quantized_accuracy_int16", "#4CAF50", "^"),
]


def create_quantization_loss_plot(studies_data, title, ylim=None):
    """
    Paper-ready comparison of quantization loss across the three strategies.

    Model architectures (bidirectional add / mul / single direction) are pooled
    into a single column per strategy, so the figure shows exactly one boxplot
    + light raw-point strip + mean diamond per strategy. A dashed line at zero
    is the "no change" reference (points below = quantized model more accurate).
    The y-axis is symlog (logarithmic, linear within ±LINTHRESH %pt) so the
    dense near-zero bulk stays readable next to large outliers; ticks are
    plain round numbers rather than log-decade labels.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys 'name' (display label = variant), 'results_data'
        (list of entries from results.json), etc. Losses are pooled across all
        studies for each strategy.
    title : str
        Basis for the saved file name and the per-dataset figure title.
    ylim : tuple of float or None
        Optional (y_low, y_high) applied to the y-axis. Pass the same value to
        comparable plots (e.g. KWS and HAR) so companion figures share an axis.
    """
    # Pool losses per strategy across ALL studies (architectures merged).
    strategy_losses = {si: [] for si in range(len(STRATEGIES))}
    for sd in studies_data:
        for si, (_label, key, _color, _marker) in enumerate(STRATEGIES):
            for rd in sd.get("results_data", []):
                fa = rd.get("test_float_accuracy", np.nan)
                qa = rd.get(key, np.nan)
                if not np.isnan(fa) and not np.isnan(qa):
                    strategy_losses[si].append(fa - qa)

    present = [si for si in range(len(STRATEGIES)) if strategy_losses[si]]
    if not present:
        print("  No quantization loss data found across any study.")
        exit(1)

    xs = {si: i for i, si in enumerate(present)}  # x centre per strategy

    fig, ax = plt.subplots(figsize=(4.4, 3.8))

    # ── Raw points (light texture), box, mean diamond per strategy ──────────
    for si in present:
        beeswarm_box(ax, fig, xs[si], strategy_losses[si],
                     STRATEGIES[si][2], STRATEGIES[si][3], show_mean=False,
                     fontsize=11)

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Reference line: points below it are cases where quantization *improved*
    # accuracy (negative loss).
    ax.axhline(0, color="#7F7F7F", linestyle="--", linewidth=0.8, zorder=3)

    # ── Axes / labels ───────────────────────────────────────────────────────
    tick_labels = [f"{STRATEGIES[si][0]}" for si in present]
    ax.set_xticks([xs[si] for si in present])
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Quantization Loss (%pt, log)", fontsize=12)
    # Symlog would default to log-decade labels (10^0, 10^1 …); show plain
    # round numbers instead, limited to the visible range so no tick is clipped.
    vmax = ylim[1] if ylim is not None else ax.get_ylim()[1]
    ax.set_yticks([t for t in (0, 5, 10, 20, 40, 80, 160, 320) if t <= vmax])
    # Override the symlog formatter, which would otherwise blank out ticks
    # that are not exact powers of ten.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    savefig(fig, title, "quant_loss", dpi=300, svg=True)
