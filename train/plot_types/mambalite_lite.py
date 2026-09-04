"""
Mamba-Lite micro comparison.

Parses the run outputs in ``experiments/mambalite-micro/*.output`` (this
work's Mamba models flashed to the ESP32-S3) and compares them against the
published Mamba-Lite micro reference figures, per dataset (HAR / KWS).

Three metrics are compared, each as one bar chart per dataset:
  * average single-inference latency (ms)
  * memory footprint (KB) as a HIERARCHICAL stacked bar per experiment, using
    the full memory profile: the bar is split into parameters (in flash),
    their RAM copy (Parameter copy, drawn in the same colour as parameters),
    variables, the model (fbs_model, non-parameter part) and others. The bar
    total is RAM + PSRAM + flash. Mamba-Lite Micro publishes no breakdown (only
    a RAM + flash total), so its bar is a single hatched bar.
  * flash model storage (bytes; converted from the KB flash total), with
    each model's bar stacked and coloured to show the parameter and the rest
    ("other") parts separately. Mamba-Lite Micro only reports a flash total,
    so its bar is drawn as a single untisplitted bar.

The RAM and flash charts share the latency chart's bar grouping: runs are
merged to one bar per model/quantization (int8 and int8-TQT averaged,
bidir-add and bidir-mul averaged into one bidirectional bar) and the
Mamba-Lite reference is inserted among the single-direction bars, with
"single direction" / "bidirectional" group labels below the axis.

Only successfully finished runs are plotted (a run is successful if the log
contains an ``INFERENCE_OK`` line together with a latency figure and a memory
summary total row). Runs that fail leave no data and are skipped.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .common import savefig, create_out_dirs

# Colour palette (matches the repository's COLORS in plot_arch_search.py).
MODEL_COLOR    = "#4C9BE8"   # this work, solid bars (default fill)
PARAM_COLOR    = "#4C9BE8"   # parameters (in fbs_model) and their RAM copy
VARIABLE_COLOR = "#9C27B0"   # variables
FBS_COLOR      = "#2E8B57"   # model (fbs_model, non-parameter part)
OTHERS_COLOR   = "#A6A6A6"   # other memory (buffers, activations, ...)
OTHER_COLOR    = "#2E8B57"   # flash "other" (non-parameter) stack segment
REF_COLOR      = "#E8834C"   # Mamba-Lite micro reference

DATASET_LABEL = {"har": "HAR", "kws": "KWS"}
DATASET_ORDER = ["har", "kws"]

LAT_RE   = re.compile(r"Average single-inference latency:\s*([\d.]+)\s*us")
TOTAL_RE = re.compile(r"\|\s*total\s*\|\s*([\d.]+)KB\s*\|\s*([\d.]+)KB\s*\|\s*([\d.]+)KB\s*\|")
# Flash breakdown line under the fbs_model entry: "|  -- parameter  | ... |  -- X KB  |"
PARAM_RE = re.compile(r"-- parameter.*?--\s*([\d.]+)\s*KB")


def _mem_row(text, name):
    """Parse one category row of the memory summary (e.g. 'fbs_model',
    'parameter_copy', 'variable', 'others') into its three column values
    (internal RAM, PSRAM, FLASH) in KB; empty cells become 0.0."""
    m = re.search(
        r"\|\s*" + name + r"\s*\|"
        r"\s*(?:(--\s*)?([\d.]+)\s*KB)?\s*\|"
        r"\s*(?:(--\s*)?([\d.]+)\s*KB)?\s*\|"
        r"\s*(?:(--\s*)?([\d.]+)\s*KB)?\s*\|", text)
    if not m:
        return None
    return [float(m.group(2 * i + 2)) if m.group(2 * i + 2) else 0.0
            for i in range(3)]

# Architecture / quantization display labels and ordering (for tidy bars).
# "single" is this work's own single-direction model, shown as "our" on the
# axis; the published reference is a float32 model, shown as Mamba-Lite-Micro.
ARCH_ORDER   = ["single", "bidir", "bidir-add", "bidir-mul"]
QUANT_ORDER  = ["int8", "int8-TQT", "int16", "float"]
ARCH_DISPLAY = {"single": "our", "bidir": "our"}
REF_LABEL    = "MambaLite-Micro\n(float32)"

# Mamba-Lite micro published reference values, per dataset: latency (ms),
# peak RAM (KB, total) and flash storage (bytes), as quoted in the brief.
REFERENCE = {
    "har": {"latency_ms": 123.4, "peak_ram_kb": 43.2, "flash_bytes": 360740},
    "kws": {"latency_ms": 1133.6, "peak_ram_kb": 230.0, "flash_bytes": 372336},
}


def _parse_stem(stem, dataset):
    """Split an output stem like 'har-bidir-add_int16' into (arch, quant)."""
    body = stem[len(dataset) + 1:]
    for suffix, quant in [("_float", "float"), ("_int16", "int16"),
                          ("_strat", "int8-TQT")]:
        if body.endswith(suffix):
            return body[:-len(suffix)], quant
    return body, "int8"


def _sort_key(entry):
    arch, quant = entry["arch"], entry["quant"]
    a = ARCH_ORDER.index(arch) if arch in ARCH_ORDER else len(ARCH_ORDER)
    q = QUANT_ORDER.index(quant) if quant in QUANT_ORDER else len(QUANT_ORDER)
    return a, q


def _merge_bars(entries, fields=("value",)):
    """Merge the runs of one bar plot: average the int8 and int8-TQT runs of
    the same architecture (one 8-bit number per model), then average the
    bidir-add and bidir-mul architectures into a single 'bidir' entry.

    ``fields`` names the numeric value(s) to average per bar (a single metric,
    or the stacked parts of a RAM / flash bar)."""
    # 1) Average the two int8 quantizations of each architecture.
    by_arch = {}
    for e in entries:
        by_arch.setdefault(e["arch"], []).append(e)
    per_arch = []
    for arch, group in by_arch.items():
        int8 = [e for e in group if e["quant"] in ("int8", "int8-TQT")]
        others = [e for e in group if e["quant"] not in ("int8", "int8-TQT")]
        if int8:
            m = dict(int8[0])
            m["quant"] = "int8"
            for f in fields:
                m[f] = float(np.mean([e[f] for e in int8]))
            per_arch.append(m)
        per_arch.extend(others)

    # 2) Average the bidir-add and bidir-mul architectures into one 'bidir'.
    out = []
    for e in per_arch:
        arch = "bidir" if e["arch"] in ("bidir-add", "bidir-mul") else e["arch"]
        e = dict(e, arch=arch)
        prev = next((o for o in out
                     if o["arch"] == arch and o["quant"] == e["quant"]), None)
        if prev is None:
            e["model"] = f"{_display_arch(arch)} ({e['quant']})"
            out.append(e)
        else:
            for f in fields:
                prev[f] = (prev[f] + e[f]) / 2.0
    out.sort(key=_sort_key)
    return out


def _display_arch(arch):
    """Architecture name as shown on the axis (e.g. 'single' -> 'our')."""
    return ARCH_DISPLAY.get(arch, arch)


def _load_runs(repo_root):
    """Parse every *.output in experiments/mambalite-micro.

    Returns a dict mapping dataset ('har'/'kws') to a list of run dicts
    (latency_ms, internal_ram_kb, psram_kb, flash_bytes, label) for successful
    runs only, sorted by architecture then quantization.
    """
    exp_dir = os.path.join(repo_root, "experiments", "mambalite-micro")
    runs = {d: [] for d in DATASET_ORDER}
    if not os.path.isdir(exp_dir):
        print(f"  WARNING: no experiments/mambalite-micro folder at {exp_dir}")
        return runs

    for fname in sorted(os.listdir(exp_dir)):
        if not fname.endswith(".output"):
            continue
        stem = fname[:-len(".output")]
        dataset = stem.split("-")[0]
        if dataset not in runs:
            continue
        text = open(os.path.join(exp_dir, fname),
                    encoding="utf-8", errors="replace").read()
        lat_m = LAT_RE.search(text)
        tot_m = TOTAL_RE.search(text)
        par_m = PARAM_RE.search(text)
        # Only count runs that actually completed: an INFERENCE line plus both
        # the latency figure and the memory-summary total row.
        if "INFERENCE" not in text or not lat_m or not tot_m:
            print(f"  skipped (no result):      {fname}")
            continue
        arch, quant = _parse_stem(stem, dataset)
        internal, psram, flash = (float(tot_m.group(1)), float(tot_m.group(2)),
                                  float(tot_m.group(3)))
        # Parameter part of the flash usage; the rest is "other". If the
        # breakdown line is missing we cannot split the bar, so fall back to
        # reporting no parameter part (all flash in "other").
        param = float(par_m.group(1)) if par_m else 0.0

        # Full memory profile: split the footprint by category (parameters and
        # their RAM copy, variables, the fbs_model blob and other memory).
        fbs, cpy, var, oth = (_mem_row(text, n) for n in
                              ("fbs_model", "parameter_copy", "variable",
                               "others"))
        fbs_flash = fbs[2] if fbs else flash
        # Parameters (the "-- parameter" flash sub-row, or the RAM copy), used
        # both for the flash split and the hierarchical breakdown.
        param = float(par_m.group(1)) if par_m else 0.0
        if not param:
            param = (cpy[1] if cpy else 0.0)
        runs[dataset].append({
            "arch": arch, "quant": quant, "model": f"{_display_arch(arch)} ({quant})",
            "latency_ms": float(lat_m.group(1)) * 1e-3,
            "internal_ram_kb": internal,
            "psram_kb": psram,
            "peak_ram_kb": internal + psram,
            "flash_bytes": flash * 1024.0,
            "flash_param_bytes": param * 1024.0,
            "flash_other_bytes": (flash - param) * 1024.0,
            # Category breakdown (KB) for the hierarchical memory figure.
            "param_flash_kb": param,
            "model_flash_kb": max(0.0, fbs_flash - param),
            "param_copy_kb": cpy[1] if cpy else 0.0,
            "variable_kb": var[1] if var else 0.0,
            "others_kb": (oth[0] + oth[1]) if oth else 0.0,
        })
        other = (flash - param)
        print(f"  parsed: {fname}  latency={runs[dataset][-1]['latency_ms']:.2f} ms | "
              f"RAM={internal:.1f}+{psram:.1f}={runs[dataset][-1]['peak_ram_kb']:.1f} KB | "
              f"flash={flash * 1024.0:.0f} B ({param * 1024.0:.0f} param + {other * 1024.0:.0f} other)")

    for d in DATASET_ORDER:
        runs[d].sort(key=_sort_key)
    return runs


GROUP_GAP = 0.75  # extra spacing between label groups (in bar-spacing units)


def _bar_positions(n, gap_before):
    """Bar centers for n bars; insert a wider gap before the given index so
    the two label groups read as visually separated clusters."""
    if gap_before is None:
        return list(range(n))
    return [i + (GROUP_GAP if i >= gap_before else 0.0) for i in range(n)]


def _tick_rotation(labels):
    """Rotate x tick labels when they do not fit between adjacent bars: with
    more than 5 bars, or with a single line longer than the bar spacing
    permits (e.g. a long one-line 'Mamba-Lite-Micro (float32)')."""
    if len(labels) > 5:
        return 45
    longest_line = max(len(line) for l in labels for line in l.split("\n"))
    if len(labels) > 1 and longest_line >= 20:
        return 45
    return 0


def _draw_bars(models, ref_value, ylabel, ytitle, filename, colors=None,
               ref_position=None, groups=None, gap_before=None):
    """One plain bar chart: this work's model bars plus the reference bar.

    ref_position: 0-based index at which the reference bar is inserted
    (default: after all model bars, e.g. to group the single-directional
    Mamba-Lite reference with this work's single-direction models).
    groups: list of (start, end, label) tuples; each draws centred text below
    the x tick labels spanning bars start..end (inclusive), e.g. to label
    the single-direction vs bidirectional groups.
    gap_before: 0-based index before which extra spacing is inserted, so the
    two label groups read as visually separated bar clusters.
    """
    labels = [m["model"] for m in models]
    values = [m["value"] for m in models]
    if colors is None:
        colors = [MODEL_COLOR] * len(models)
    colors = list(colors)
    hatch = [None] * len(models)
    if ref_position is None:
        ref_position = len(models)
    labels.insert(ref_position, REF_LABEL)
    values.insert(ref_position, ref_value)
    colors.insert(ref_position, REF_COLOR)
    hatch.insert(ref_position, "//")   # distinguish the reference in B/W

    n = len(values)
    positions = _bar_positions(n, gap_before)
    rotation = _tick_rotation(labels)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * n), 6))
    ax.bar(positions, values, color=colors, edgecolor="white", linewidth=0.5,
           hatch=hatch, width=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    group_y = -0.24 if rotation else -0.13
    for start, end, label in groups or []:
        ax.text((positions[start] + positions[end]) / 2.0, group_y, label,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9)
    ax.legend(handles=[
        Patch(facecolor=MODEL_COLOR, edgecolor="white", label="This work (Mamba)"),
        Patch(facecolor=REF_COLOR, hatch="//", edgecolor="white",
              label=REF_LABEL),
    ], fontsize=9, framealpha=1.0)
    savefig(fig, ytitle, f"mambalite_{filename}", dpi=300)


def _draw_hier_ram(models, dataset, ytitle, ref_position=None,
                    groups=None, gap_before=None):
    """Hierarchical memory bar chart: ONE stacked bar per experiment, using the
    full memory-profile breakdown (per category and memory type). Each bar is
    segmented bottom-to-top into:
      parameters (the model's parameter blob in flash)
      parameter copy (the RAM copy, drawn in the same colour as parameters)
      variables
      model (fbs_model, the non-parameter part of the flash model blob)
      others (remaining memory)
    The total bar height is therefore RAM + PSRAM + flash. The Mamba-Lite
    reference only publishes a combined RAM + flash total, so its bar is a
    single hatched bar. ref_position: model index at which the reference is
    inserted (default: at the end of the model bars) so it groups with this
    work's single-direction models; groups/gap_before label and separate the
    single-direction and bidirectional bar clusters."""
    # Segments, bottom-to-top, with (key, label, colour); parameters and their
    # RAM copy share a colour ("the copy shown the same way").
    segments = [
        ("param_flash_kb", "Parameters", PARAM_COLOR),
        ("param_copy_kb", "Parameter copy", PARAM_COLOR),
        ("variable_kb", "Variables", VARIABLE_COLOR),
        ("model_flash_kb", "Model (fbs_model)", FBS_COLOR),
        ("others_kb", "Others", OTHERS_COLOR),
    ]
    labels = [m["model"] for m in models]
    vals = {key: [m[key] for m in models] for key, _, _ in segments}
    if ref_position is None:
        ref_position = len(models)
    labels.insert(ref_position, REF_LABEL)
    # Reference total footprint = RAM + flash (it has no category breakdown).
    ref_total = REFERENCE[dataset]["peak_ram_kb"] \
        + REFERENCE[dataset]["flash_bytes"] / 1024.0
    for key in vals:
        vals[key].insert(ref_position, 0.0)

    n = len(labels)
    positions = _bar_positions(n, gap_before)
    rotation = _tick_rotation(labels)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * n), 6))

    model_idx = [i for i in range(n) if i != ref_position]
    bottoms = [0.0] * n
    for key, _label, color in segments:
        parts = [vals[key][i] for i in model_idx]
        ax.bar([positions[i] for i in model_idx], parts,
               bottom=[bottoms[i] for i in model_idx], color=color,
               edgecolor="white", linewidth=0.5, width=0.8)
        for i, v in zip(model_idx, parts):
            bottoms[i] += v

    # Mamba-Lite reference: single hatched bar at its total footprint.
    ax.bar([positions[ref_position]], [ref_total], color=REF_COLOR,
           edgecolor="white", linewidth=0.5, width=0.8, hatch="//")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center", fontsize=9)
    ax.set_ylabel("Memory (KB)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    group_y = -0.24 if rotation else -0.13
    for start, end, label in groups or []:
        ax.text((positions[start] + positions[end]) / 2.0, group_y, label,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9)
    handles = [Patch(facecolor=color, edgecolor="white", label=label)
               for _, label, color in segments]
    handles.append(Patch(facecolor=REF_COLOR, hatch="//", edgecolor="white",
                         label=REF_LABEL))
    ax.legend(handles=handles, fontsize=9, framealpha=1.0)
    savefig(fig, ytitle, f"mambalite_peakram_{dataset}", dpi=300)


def _draw_stacked_flash(models, dataset, ytitle, ref_position=None,
                        groups=None, gap_before=None):
    """Flash-storage bars stacked by content: each model's parameter and
    "other" (everything else, e.g. program code) parts are stacked and
    coloured separately. The Mamba-Lite reference has no such breakdown (only
    a flash total is published), so its bar stays a single hatched bar,
    inserted at ref_position (default: at the end of the model bars) so it
    groups with this work's single-direction models; groups/gap_before label
    and separate the single-direction and bidirectional bar clusters."""
    labels = [m["model"] for m in models]
    params = [m["flash_param_bytes"] for m in models]
    others = [m["flash_other_bytes"] for m in models]
    if ref_position is None:
        ref_position = len(models)
    labels.insert(ref_position, REF_LABEL)
    params.insert(ref_position, 0.0)
    others.insert(ref_position, REFERENCE[dataset]["flash_bytes"])

    n = len(labels)
    positions = _bar_positions(n, gap_before)
    rotation = _tick_rotation(labels)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * n), 6))

    model_idx = [i for i in range(n) if i != ref_position]
    # This work's models: stacked parameters (bottom) + other (top).
    ax.bar([positions[i] for i in model_idx], [params[i] for i in model_idx],
           color=PARAM_COLOR, edgecolor="white", linewidth=0.5, width=0.8)
    ax.bar([positions[i] for i in model_idx], [others[i] for i in model_idx],
           bottom=[params[i] for i in model_idx], color=OTHER_COLOR,
           edgecolor="white", linewidth=0.5, width=0.8)

    # Mamba-Lite reference: single bar (no parameter/other breakdown), hashed.
    ax.bar([positions[ref_position]], [others[ref_position]], color=REF_COLOR,
           edgecolor="white", linewidth=0.5, width=0.8, hatch="//")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center", fontsize=9)
    ax.set_ylabel("Flash storage (bytes)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    group_y = -0.24 if rotation else -0.13
    for start, end, label in groups or []:
        ax.text((positions[start] + positions[end]) / 2.0, group_y, label,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9)
    handles = [
        Patch(facecolor=PARAM_COLOR, edgecolor="white", label="Parameters"),
        Patch(facecolor=OTHER_COLOR, edgecolor="white", label="Other"),
        Patch(facecolor=REF_COLOR, hatch="//", edgecolor="white",
              label=REF_LABEL),
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=1.0)
    savefig(fig, ytitle, f"mambalite_flash_{dataset}", dpi=300)


def create_mambalite_lite_plot(repo_root, title=None):
    """Generate all Mamba-Lite comparison figures (latency / peak RAM / flash
    for HAR and KWS) from the raw run outputs."""
    create_out_dirs()
    print("Parsing mambalite-micro run outputs ...")
    runs = _load_runs(repo_root)

    any_models = False
    for dataset in DATASET_ORDER:
        models = runs[dataset]
        ref = REFERENCE[dataset]
        dl = DATASET_LABEL[dataset]
        if not models:
            print(f"  No successful Mamba runs to plot for {dl} (reference only).")
        else:
            any_models = True

        # Latency (quantization levels and bidir variants averaged per bar).
        # The Mamba-Lite reference is single-directional, so it is inserted
        # right after this work's single-direction bars (3rd position) and the
        # two groups are labelled below the axis.
        merged = _merge_bars([dict(m, value=m["latency_ms"]) for m in models])
        n_single = sum(1 for m in merged if m["arch"] == "single")
        groups = None
        if merged and n_single < len(merged):
            groups = [
                (0, n_single, "single direction"),
                (n_single + 1, len(merged), "bidirectional"),
            ]
        _draw_bars(
            merged, ref["latency_ms"],
            "Latency (ms)", f"Latency vs Mamba-Lite micro ({dl})",
            f"latency_{dataset}",
            ref_position=n_single if merged else None, groups=groups,
            gap_before=(n_single + 1) if groups else None)
        # Memory: hierarchical stacked bars (parameters, parameter copy,
        # variables, model/fbs_model, others) using the full memory profile,
        # with the same bar grouping as the latency chart.
        merged_ram = _merge_bars(models, fields=(
            "internal_ram_kb", "psram_kb", "flash_bytes",
            "param_flash_kb", "param_copy_kb", "variable_kb",
            "model_flash_kb", "others_kb"))
        _draw_hier_ram(
            merged_ram, dataset, f"Memory vs Mamba-Lite micro ({dl})",
            ref_position=n_single if merged_ram else None, groups=groups,
            gap_before=(n_single + 1) if groups else None)
        # Flash storage in bytes (parameter / other parts stacked), with the
        # same bar grouping as the latency chart.
        merged_flash = _merge_bars(models, fields=("flash_param_bytes", "flash_other_bytes"))
        _draw_stacked_flash(
            merged_flash, dataset, f"Flash size vs Mamba-Lite micro ({dl})",
            ref_position=n_single if merged_flash else None, groups=groups,
            gap_before=(n_single + 1) if groups else None)

    if not any_models:
        print("  No successful runs in experiments/mambalite-micro; "
              "figures show the Mamba-Lite reference only.")
