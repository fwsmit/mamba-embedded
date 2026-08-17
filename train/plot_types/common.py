import re
import os

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
