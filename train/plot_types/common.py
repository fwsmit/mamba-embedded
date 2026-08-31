import math
import re
import os

import numpy as np

OUT_DIR = "figures"
PDF_DIR = os.path.join(OUT_DIR, "pdf")

FIG_DPI      = 150

# (--size, --quantization) → (results.json accuracy field, y-axis label),
# shared by the param_accuracy and mcu_pareto plots.
ACCURACY_FIELDS = {
    (32, "no"): "test_float_accuracy",
    (16, "percent"): "test_quantized_accuracy_int16",
    (8, "percent"): "test_quantized_accuracy",
    (8, "tqt"): "test_quantized_accuracy_strat",
}

# Validation-set counterparts of the test fields above; used to select the
# highlighted models from the Pareto front (plotting stays on the test fields).
SELECTION_FIELDS = {
    (32, "no"): "float_accuracy",
    (16, "percent"): "quantized_accuracy_int16",
    (8, "percent"): "quantized_accuracy",
    (8, "tqt"): "quantized_accuracy_strat",
}

ACCURACY_LABELS = {
    (32, "no"): "Float accuracy (%)",
    (16, "percent"): "Quantized accuracy (int16, %)",
    (8, "percent"): "Quantized accuracy (int8, %)",
    (8, "tqt"): "Quantized accuracy (int8, TQT, %)",
}


def resolve_accuracy(size, quantization):
    """Map a (--size, --quantization) pair to the test results.json accuracy
    field, its validation-set counterpart (used to select highlighted models),
    and the y-axis label. Shared by the param_accuracy and mcu_pareto plots."""
    key = (size, quantization)
    if key not in ACCURACY_FIELDS:
        supported = ", ".join(
            f"--size {s} --quantization {q}" for s, q in ACCURACY_FIELDS)
        raise ValueError(
            f"Unsupported combination --size {size} --quantization {quantization}. "
            f"Supported combinations: {supported}.")
    return ACCURACY_FIELDS[key], SELECTION_FIELDS[key], ACCURACY_LABELS[key]


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = re.sub(r"[ _]+", "-", s)
    return s


def create_out_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def fig_path(name):
    return os.path.join(OUT_DIR, name)
def fig_pdf_path(name):
    return os.path.join(PDF_DIR, name)


def savefig(fig, title, filename, dpi=None, svg=False):
    dpi = FIG_DPI if dpi is None else dpi
    fig.tight_layout()
    slug = slugify(title)
    fig_path_png = fig_path(f"{filename}_{slug}.png")
    fig_path_pdf = fig_pdf_path(f"{filename}_{slug}.pdf")
    fig.savefig(fig_path_png, dpi=dpi)
    fig.savefig(fig_path_pdf)
    if svg:
        fig_path_svg = fig_path(f"{filename}_{slug}.svg")
        fig.savefig(fig_path_svg)
    print(f"Saved figures to {fig_path_png}")
    print()


# ── Beeswarm / box-column helpers ─────────────────────────────────────────────
# Shared by the quantization_loss and val_test_gap plots: a column of light
# raw points fanned out sideways (beeswarm), a boxplot and a black mean
# diamond with a numeric label.

# Symlog linearity threshold (%pt). Values within ±LINTHRESH are drawn on a
# linear scale, everything outside on a log scale — so the dense near-zero mass
# stays readable instead of being compressed into a sliver by the high outlier.
LINTHRESH = 1.0

# Vertical (in symlog space) distance beyond which two points occupy a
# different y-level and therefore do not need horizontal separation.
SWARM_YTOL = 0.25

# Box width and horizontal spacing between columns.
BOX_WIDTH = 0.5
COL_SPACING = 1.0

# Horizontal jitter band (as a fraction of COL_SPACING) over which the raw
# points are fanned out, so the strip reads as light texture behind the box
# rather than a tight vertical smear.
JITTER = 0.22


def symlog(v):
    """Monotonic transform approximating matplotlib's symlog spacing, used for
    overlap detection in the beeswarm. Linear near 0, logarithmic further out."""
    if v == 0:
        return 0.0
    return math.copysign(math.log10(1.0 + abs(v) / LINTHRESH), v)


def beeswarm_offsets(values, seed=7):
    """Horizontal offsets that fan points sharing a similar y-level out sideways
    instead of stacking them into an opaque blob at identical values."""
    rng = np.random.default_rng(seed)
    t = np.array([symlog(v) for v in values])
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


def pt_to_data(ax, fig, pt):
    """Convert a point offset to x-data units for the current axes geometry."""
    renderer = fig.canvas.get_renderer()
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent(renderer=renderer)
    px_per_unit = bbox.width / (x1 - x0)
    return pt / 72.0 * fig.dpi / px_per_unit


def beeswarm_box(ax, fig, x, values, color, marker, box_width=BOX_WIDTH,
                show_mean=True):
    """Draw one column of light raw points + boxplot + black mean diamond with
    a numeric label. Returns (mean, median, q1, q3, label). Pass show_mean=False
    to skip the mean diamond and its label."""
    values = np.asarray(values, dtype=float)
    offs = beeswarm_offsets(values)
    ax.scatter(x + offs, values, s=16, alpha=0.5, color=color, marker=marker,
               edgecolors="none", zorder=1)
    ax.boxplot([values], positions=[x], widths=box_width, showfliers=False,
               whis=1.5, zorder=2, patch_artist=True,
               boxprops=dict(facecolor="#BDBDBD", alpha=0.4,
                             edgecolor="#616161"),
               whiskerprops=dict(color="#616161"),
               capprops=dict(color="#616161"),
               medianprops=dict(color="#424242", linewidth=1.0))
    mean = float(np.mean(values))
    med = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    label = None
    if show_mean:
        outlier = abs(mean - med) > 1.5 * (q3 - q1) + 1e-9
        ax.scatter([x], [mean], s=30, marker="D", color="black", zorder=4)
        label = f"{mean:.2f}{'*' if outlier else ''}"
        bbox = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1)
        # Mean label: to the right of the diamond by default; if that fixed
        # offset would land on the box edge (narrow columns), nudge vertically
        # (above the box) instead so it never collides with the box.
        if x + pt_to_data(ax, fig, 6.0) < x + box_width / 2:
            ax.annotate(label, (x, mean), textcoords="offset points",
                        xytext=(0, 5), ha="center", va="bottom",
                        fontsize=8, zorder=6, color="black", bbox=bbox)
        else:
            ax.annotate(label, (x, mean), textcoords="offset points",
                        xytext=(6, 0), ha="left", va="center",
                        fontsize=8, zorder=6, color="black", bbox=bbox)
    return mean, med, q1, q3, label
