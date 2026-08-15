import re
import os

OUT_DIR = "figures"
PDF_DIR = os.path.join(OUT_DIR, "pdf")

FIG_DPI      = 150

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
