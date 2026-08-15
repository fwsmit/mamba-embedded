import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import savefig

# Strategies: (display label, result-field key, colour, marker). The colour
# mapping (blue/purple/green) is kept consistent across all plots; distinct
# markers add a grayscale / colour-blind-safe secondary channel.
STRATEGIES = [
    ("8-bit (prc)",    "test_quantized_accuracy",       "#4C9BE8", "o"),
    ("8-bit (kl-tqt)", "test_quantized_accuracy_strat", "#9C27B0", "s"),
    ("16-bit (prc)",   "test_quantized_accuracy_int16", "#4CAF50", "^"),
]

# Symlog linearity threshold (%pt). Values within ±LINTHRESH are drawn on a
# linear scale, everything outside on a log scale — so the dense near-zero mass
# stays readable instead of being compressed into a sliver by the high outlier.
LINTHRESH = 1.0

# Vertical (in symlog space) distance beyond which two points occupy a
# different y-level and therefore do not need horizontal separation.
SWARM_YTOL = 0.25

# Box width and horizontal spacing between the three strategy columns.
BOX_WIDTH = 0.5
COL_SPACING = 1.0

# Horizontal jitter band (as a fraction of COL_SPACING) over which the raw
# points are fanned out, so the strip reads as light texture behind the box
# rather than a tight vertical smear.
JITTER = 0.22


def _symlog(v):
    """Monotonic transform approximating matplotlib's symlog spacing, used for
    overlap detection in the beeswarm. Linear near 0, logarithmic further out."""
    if v == 0:
        return 0.0
    return math.copysign(math.log10(1.0 + abs(v) / LINTHRESH), v)


def _beeswarm_offsets(values, seed=7):
    """Horizontal offsets that fan points sharing a similar y-level out sideways
    instead of stacking them into an opaque blob at identical values."""
    rng = np.random.default_rng(seed)
    t = np.array([_symlog(v) for v in values])
    order = np.argsort(t)
    offsets = np.zeros(len(values))
    placed = []
    candidates = np.linspace(-JITTER * COL_SPACING, JITTER * COL_SPACING, 89)
    for k in order:
        tk = t[k]
        cand = candidates.copy()
        rng.shuffle(cand)
        chosen = 0.0
        for c in cand:
            if all(abs(pt - tk) >= SWARM_YTOL or abs(po - c) >= 0.015
                   for (pt, po) in placed):
                chosen = c
                break
        offsets[k] = chosen
        placed.append((tk, chosen))
    return offsets


def create_quantization_loss_plot(studies_data, title, ylim=None):
    """
    Paper-ready comparison of quantization loss across the three strategies.

    Model architectures (bidirectional add / mul / single direction) are pooled
    into a single column per strategy, so the figure shows exactly one boxplot
    + light raw-point strip + mean diamond per strategy. A dashed line at zero
    is the "no change" reference (points below = quantized model more accurate).
    A symlog y-axis (linear near 0) keeps both the dense near-zero bulk and the
    large outliers readable.

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

    fig, ax = plt.subplots(figsize=(3.6, 3.3))

    # ── Raw points (light texture), box, mean diamond per strategy ──────────
    for si in present:
        losses = np.array(strategy_losses[si])
        c = xs[si]
        offs = _beeswarm_offsets(losses)
        ax.scatter(c + offs, losses, s=16, alpha=0.5,
                   color=STRATEGIES[si][2], marker=STRATEGIES[si][3],
                   edgecolors="none", zorder=1)
        ax.boxplot([losses], positions=[c], widths=BOX_WIDTH, showfliers=False,
                   whis=1.5, zorder=2, patch_artist=True,
                   boxprops=dict(facecolor="#BDBDBD", alpha=0.4,
                                 edgecolor="#616161"),
                   whiskerprops=dict(color="#616161"),
                   capprops=dict(color="#616161"),
                   medianprops=dict(color="#424242", linewidth=1.0))
        mean = float(np.mean(losses))
        med = float(np.median(losses))
        q1, q3 = np.percentile(losses, [25, 75])
        outlier = abs(mean - med) > 1.5 * (q3 - q1) + 1e-9
        ax.scatter([c], [mean], s=30, marker="D", color="black", zorder=4)
        label = f"{mean:.2f}{'*' if outlier else ''}"
        bbox = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1)
        # Mean label: to the right of the diamond by default; if that fixed
        # offset would land on the box edge (narrow columns), nudge vertically
        # (above the box) instead so it never collides with the box.
        if c + _pt_to_data(ax, fig, 6.0) < c + BOX_WIDTH / 2:
            ax.annotate(label, (c, mean), textcoords="offset points",
                        xytext=(0, 5), ha="center", va="bottom",
                        fontsize=8, zorder=6, color="black", bbox=bbox)
        else:
            ax.annotate(label, (c, mean), textcoords="offset points",
                        xytext=(6, 0), ha="left", va="center",
                        fontsize=8, zorder=6, color="black", bbox=bbox)

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Reference line: points below it are cases where quantization *improved*
    # accuracy (negative loss).
    ax.axhline(0, color="#7F7F7F", linestyle="--", linewidth=0.8, zorder=3)

    # ── Axes / labels ───────────────────────────────────────────────────────
    tick_labels = [f"{STRATEGIES[si][0]}\n(n={len(strategy_losses[si])})"
                   for si in present]
    ax.set_xticks([xs[si] for si in present])
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Quantization Loss (%pt)", fontsize=9)
    ax.set_yticks([0, 1, 10, 100])
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Per-chart title: tells the reader which dataset the figure is for.
    dataset = next((d for d in ("har", "kws") if d in title.lower()), "")
    prefix = dataset.upper() if dataset else "Quantization Loss"
    ax.set_title(f"{prefix}: Quantization Loss by Strategy",
                 fontsize=10, fontweight="bold")

    # ── Legend: one entry per strategy, laid out horizontally below the plot.
    handles = [Line2D([0], [0], marker=STRATEGIES[si][3], color="w",
                      markerfacecolor=STRATEGIES[si][2], markersize=6,
                      label=STRATEGIES[si][0]) for si in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=len(present), frameon=False, fontsize=8, handlelength=2)

    savefig(fig, title, "quant_loss", dpi=300, svg=True)


def _pt_to_data(ax, fig, pt):
    """Convert a point offset to x-data units for the current axes geometry."""
    renderer = fig.canvas.get_renderer()
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent(renderer=renderer)
    px_per_unit = bbox.width / (x1 - x0)
    return pt / 72.0 * fig.dpi / px_per_unit
