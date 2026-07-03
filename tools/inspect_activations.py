#!/usr/bin/env python3
"""
inspect_activations.py — Analyse intermediate tensor activations from an ONNX
model to diagnose poor quantisation.

The tool:
1. Loads an ONNX model and adds every intermediate node output as a graph output.
2. Loads validation data (HAR or KWS, inferred from the input shape).
3. Runs ONNX Runtime inference on a small batch of samples.
4. Reports per-tensor statistics (min, max, mean, std, dynamic range, etc.)
   to identify tensors with wide internal ranges — a sign that per-tensor
   quantisation will be lossy.

Usage:
    conda activate torch-pascal
    python tools/inspect_activations.py path/to/model.onnx
    python tools/inspect_activations.py path/to/model.onnx --samples 50
    python tools/inspect_activations.py path/to/model.onnx --threshold 10 --samples 100
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Add the project root so we can import train.data
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.data import load_har_data, load_speechcommands_data  # noqa: E402


# ---------------------------------------------------------------------------
# Known input shapes → dataset name
# ---------------------------------------------------------------------------
INPUT_SHAPE_MAP = {
    (1, 10, 57): "har",
    (1, 49, 40): "kws",
}


def infer_dataset(onnx_path: Path) -> str:
    """Infer dataset name from the ONNX model's input shape."""
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
    if shape not in INPUT_SHAPE_MAP:
        print(f"ERROR: Unknown input shape {shape}.", file=sys.stderr)
        print(f"  Known: {list(INPUT_SHAPE_MAP.keys())}", file=sys.stderr)
        sys.exit(1)
    return INPUT_SHAPE_MAP[shape]


# ---------------------------------------------------------------------------
# ONNX graph augmentation: add every intermediate node output
# ---------------------------------------------------------------------------
_DISALLOWED_OUTPUTS = frozenset({
    # Initializer-like names that are constants, not activations
})


def add_intermediate_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add every unique intermediate node output as a graph output.

    Returns a new ModelProto with the extra outputs. The original is unmodified.
    """
    # Collect existing output names (so we don't re-add)
    existing_outputs = {o.name for o in model.graph.output}
    existing_values = {v.name for v in model.graph.value_info}
    # Also consider initializers and graph inputs
    known_names = set()
    for init in model.graph.initializer:
        known_names.add(init.name)
    for inp in model.graph.input:
        known_names.add(inp.name)
    known_names.update(existing_outputs)

    # Collect all unique intermediate outputs from nodes
    intermediate_names: list[str] = []
    seen = set()
    for node in model.graph.node:
        for output_name in node.output:
            if output_name and output_name not in seen and output_name not in known_names:
                seen.add(output_name)
                intermediate_names.append(output_name)

    # Create value_info entries for the new outputs
    # (ONNX Runtime needs type/shape info for some, but often works without)
    from onnx import helper, TensorProto

    new_outputs = []
    for name in intermediate_names:
        # Try to get shape from existing value_info
        vi = None
        for existing_vi in model.graph.value_info:
            if existing_vi.name == name:
                vi = existing_vi
                break

        if vi is not None:
            new_outputs.append(vi)
        else:
            # Create a minimal value_info — dtype float, no shape constraint
            new_outputs.append(
                helper.make_tensor_value_info(
                    name, TensorProto.FLOAT, None
                )
            )

    # Create a new graph with all intermediate outputs appended
    new_graph = helper.make_graph(
        nodes=list(model.graph.node),
        name=model.graph.name,
        inputs=list(model.graph.input),
        outputs=list(model.graph.output) + new_outputs,
        initializer=list(model.graph.initializer),
        value_info=list(model.graph.value_info),
    )

    new_model = helper.make_model(
        new_graph,
        opset_imports=model.opset_import,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        ir_version=model.ir_version,
    )
    new_model.ir_version = model.ir_version
    return new_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_val_data(dataset: str, n_samples: int):
    """Load up to *n_samples* validation samples from the given dataset."""
    data_dir = Path.home() / "Datasets"
    if dataset == "har":
        ds = load_har_data(data_dir, split="val")
    else:
        ds = load_speechcommands_data(str(data_dir), split="val")
    n = min(n_samples, len(ds))
    # Take first N samples
    xs, ys = [], []
    for i in range(n):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs, axis=0), np.array(ys)


# ---------------------------------------------------------------------------
# Activation analysis
# ---------------------------------------------------------------------------
def analyse_tensor(name: str, values: np.ndarray) -> dict:
    """Compute per-tensor statistics.

    Returns a dict with keys:
        name, shape, min, max, mean, std,
        dynamic_range (max - min),
        range_ratio (max / min_abs),
        cv (std / abs(mean)),
        n_zeros, n_near_zero (|x| < 1e-6),
        per_channel — per-channel diagnostics if the tensor is 2D+
    """
    info: dict = {
        "name": name,
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }

    vmin = info["min"]
    vmax = info["max"]
    info["dynamic_range"] = vmax - vmin

    # Range ratio: how many "bins" across zero
    min_abs = min(abs(vmin), abs(vmax))
    if vmin < 0 and vmax > 0:
        # crosses zero — the whole range matters
        info["range_ratio"] = (vmax - vmin) / max(min_abs, 1e-12)
    else:
        info["range_ratio"] = vmax / max(vmin, 1e-12) if vmin != 0 else float("inf")

    # Coefficient of variation
    mean_abs = abs(info["mean"])
    info["cv"] = info["std"] / max(mean_abs, 1e-12)

    # Near-zero counts
    flat = values.flatten()
    info["n_zeros"] = int((flat == 0).sum())
    info["n_near_zero"] = int((np.abs(flat) < 1e-6).sum())
    info["total_elts"] = flat.size
    zero_frac = info["n_near_zero"] / max(info["total_elts"], 1)
    info["near_zero_fraction"] = zero_frac

    # Per-channel analysis for >= 2D tensors
    # Look at the last dimension (channel dimension for activations)
    if values.ndim >= 2:
        # For (B, T, C) or (B, C, T), analyse along C
        # Try to find the C dimension — usually the last one
        ch_data = values  # shape (..., C)
        # axes to reduce: all except last
        reduce_axes = tuple(range(values.ndim - 1))
        ch_min = values.min(axis=reduce_axes)  # (C,)
        ch_max = values.max(axis=reduce_axes)
        ch_mean = values.mean(axis=reduce_axes)
        ch_std = values.std(axis=reduce_axes)
        ch_range = ch_max - ch_min

        info["channels"] = len(ch_min)
        info["ch_min_min"] = float(ch_min.min())
        info["ch_min_max"] = float(ch_min.max())
        info["ch_max_min"] = float(ch_max.min())
        info["ch_max_max"] = float(ch_max.max())
        info["ch_range_min"] = float(ch_range.min())
        info["ch_range_max"] = float(ch_range.max())

        # How much do channel ranges differ?
        # If one channel spans [-10, 10] and another spans [-0.1, 0.1],
        # per-tensor quantisation will waste bits on the quiet channel.
        info["ch_range_ratio"] = float(ch_range.max() / max(ch_range.min(), 1e-12))

        # Per-channel range imbalance
        if ch_range.max() > 0:
            info["ch_range_imbalance"] = float(ch_range.max() / ch_range.mean())
        else:
            info["ch_range_imbalance"] = 1.0

        # If the tensor crosses zero, compute the max absolute value per channel
        ch_absmax = np.maximum(np.abs(ch_min), np.abs(ch_max))
        info["ch_absmax_min"] = float(ch_absmax.min())
        info["ch_absmax_max"] = float(ch_absmax.max())
        info["ch_absmax_ratio"] = float(
            ch_absmax.max() / max(ch_absmax.min(), 1e-12)
        )
    else:
        info["channels"] = 1
        info["ch_range_ratio"] = 1.0
        info["ch_range_imbalance"] = 1.0
        info["ch_absmax_ratio"] = 1.0

    return info


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def c(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}" if sys.stdout.isatty() else text


def print_report(all_stats: list[dict], threshold: float = 5.0):
    """Print a formatted report.

    *threshold* is the ``ch_range_ratio`` or ``range_ratio`` above which a tensor
    is flagged as a concern.
    """
    # Sort by channel range ratio descending (most imbalanced first)
    sorted_stats = sorted(all_stats, key=lambda s: s["ch_range_ratio"], reverse=True)

    # Separate concerning vs OK
    concerning = [s for s in sorted_stats if
                  s["ch_range_ratio"] > threshold or
                  s["range_ratio"] > threshold * 2 or
                  s["ch_absmax_ratio"] > threshold]

    ok_tensors = [s for s in sorted_stats if s not in concerning]

    print()
    print(c("=" * 78, ANSI_BOLD))
    print(c("  INTERMEDIATE ACTIVATION ANALYSIS", ANSI_BOLD))
    print(c("=" * 78, ANSI_BOLD))
    print(f"  Total intermediate tensors : {len(all_stats)}")
    print(f"  Flagged as concerning      : {len(concerning)}  "
          f"(ch_range_ratio > {threshold} | range_ratio > {threshold*2} | ch_absmax_ratio > {threshold})")
    print()

    # ── Concerning tensors ─────────────────────────────────────────────
    if concerning:
        print(c("  ⚠  TENSORS WITH POTENTIAL QUANTISATION ISSUES", ANSI_YELLOW + ANSI_BOLD))
        print(c("  " + "-" * 78, ANSI_YELLOW))
        for s in concerning:
            shape_str = "×".join(str(d) if d else "?" for d in s["shape"])
            line = (
                f"  {s['name']:<45s}  [{shape_str:<16s}]\n"
                f"    range   [{s['min']:<12.6g}, {s['max']:<12.6g}]  "
                f"mean={s['mean']:<12.6g}  σ={s['std']:<12.6g}\n"
                f"    dynamic_range={s['dynamic_range']:<12.6g}  "
                f"range_ratio={s['range_ratio']:<10.4g}\n"
                f"    near_zero={s['near_zero_fraction']*100:<5.1f}%  "
                f"CV={s['cv']:<10.4g}\n"
                f"    ch_range=[{s['ch_range_min']:<12.6g}, {s['ch_range_max']:<12.6g}]  "
                f"ch_range_ratio={s['ch_range_ratio']:<10.4g}\n"
                f"    ch_absmax=[{s['ch_absmax_min']:<12.6g}, {s['ch_absmax_max']:<12.6g}]  "
                f"ch_absmax_ratio={s['ch_absmax_ratio']:<10.4g}\n"
            )
            # Colour code severity
            if s["ch_range_ratio"] > threshold * 4:
                colour = ANSI_RED
            elif s["ch_range_ratio"] > threshold * 2:
                colour = ANSI_YELLOW
            else:
                colour = ANSI_CYAN
            print(c(line, colour))
        print()

    # ── Top-10 OK summary ─────────────────────────────────────────────
    if ok_tensors:
        print(c("  ✓ REMAINING TENSORS (top 10 by ch_range_ratio)", ANSI_GREEN if not concerning else ""))
        print(c("  " + "-" * 78, ANSI_GREEN if not concerning else ""))
        for s in ok_tensors[:10]:
            shape_str = "×".join(str(d) if d else "?" for d in s["shape"])
            print(
                f"  {s['name']:<45s}  [{shape_str:<16s}]  "
                f"range=[{s['min']:<10.4g},{s['max']:<10.4g}]  "
                f"ch_range_ratio={s['ch_range_ratio']:<8.3g}  "
                f"near_zero={s['near_zero_fraction']*100:<5.1f}%"
            )
        if len(ok_tensors) > 10:
            print(f"  ... and {len(ok_tensors) - 10} more")
        print()


# ---------------------------------------------------------------------------
# Summary of key findings
# ---------------------------------------------------------------------------
def print_summary(all_stats: list[dict]):
    """Print a compact one-line-per-tensor table for easy grepping/plotting."""
    print()
    print(c("=" * 78, ANSI_BOLD))
    print(c("  PER-TENSOR SUMMARY TABLE", ANSI_BOLD))
    print(c("=" * 78, ANSI_BOLD))
    print(f"  {'name':<45s} {'shape':<16s} {'min':>10s} {'max':>10s} "
          f"{'ch_range_r':>10s} {'near_zero':>9s}")
    print("  " + "-" * 78)
    for s in sorted(all_stats, key=lambda x: x["ch_range_ratio"], reverse=True):
        shape_str = "×".join(str(d) if d else "?" for d in s["shape"])
        name_trunc = s["name"][:44]
        print(f"  {name_trunc:<45s} {shape_str:<16s} {s['min']:>10.4g} {s['max']:>10.4g} "
              f"{s['ch_range_ratio']:>10.4g} {s['near_zero_fraction']*100:>8.2f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse intermediate ONNX activations for quantisation quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("model", type=Path, help="Path to the ONNX model file")
    p.add_argument(
        "--samples", type=int, default=32,
        help="Number of validation samples to run (default: 32)",
    )
    p.add_argument(
        "--threshold", type=float, default=5.0,
        help="ch_range_ratio / absmax_ratio threshold for flagging (default: 5.0)",
    )
    p.add_argument(
        "--summary-only", action="store_true",
        help="Only print the compact summary table, skip detailed report",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Dump results as JSON to stdout",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading ONNX model: {model_path}")

    dataset = infer_dataset(model_path)
    print(f"  Inferred dataset   : {dataset}")

    model = onnx.load(str(model_path))
    n_nodes = len(model.graph.node)
    print(f"  Nodes in graph     : {n_nodes}")

    # Augment the graph with intermediate outputs
    print("  Adding intermediate outputs to graph ...")
    augmented_model = add_intermediate_outputs(model)
    n_extra = len(augmented_model.graph.output) - len(model.graph.output)
    print(f"  Added {n_extra} intermediate outputs")

    # Save to a temp file (ONNX Runtime needs a file, not in-memory model)
    tmp_path = model_path.with_suffix(".tmp-augmented.onnx")
    onnx.save(augmented_model, str(tmp_path))

    # Load validation data
    print(f"  Loading {args.samples} validation samples ...")
    xs, _ = load_val_data(dataset, args.samples)
    print(f"  Data shape: {xs.shape}")

    # Run ONNX Runtime inference
    print("  Running inference ...")
    ort_sess = ort.InferenceSession(str(tmp_path))

    # Get output names (first output is the original, rest are intermediate)
    output_names = [o.name for o in augmented_model.graph.output]
    print(f"  Output tensors     : {len(output_names)}")

    # Run inference on all samples (batch=1 to get per-sample activations)
    all_intermediates: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in range(len(xs)):
        sample = xs[i:i + 1]  # keep batch dim
        outputs = ort_sess.run(output_names, {"input": sample})
        for name, arr in zip(output_names, outputs):
            all_intermediates[name].append(arr)

    # Clean up temp file
    tmp_path.unlink()

    print("  Analysing activations ...")

    # Aggregate: concatenate along batch dimension
    all_stats = []
    for name, arrays in all_intermediates.items():
        if name == "output":
            continue  # skip the final output
        try:
            cat = np.concatenate(arrays, axis=0)
        except Exception:
            # Some may have variable shapes (e.g. scalar outputs) — skip
            continue
        if cat.size == 0:
            continue
        info = analyse_tensor(name, cat)
        all_stats.append(info)

    all_stats.sort(key=lambda s: s["ch_range_ratio"], reverse=True)

    # Output
    if args.json:
        import json
        print(json.dumps(all_stats, indent=2, default=str))
    elif args.summary_only:
        print_summary(all_stats)
    else:
        print_report(all_stats, threshold=args.threshold)
        print_summary(all_stats)

    # Print key takeaway
    n_flagged = sum(
        1 for s in all_stats
        if s["ch_range_ratio"] > args.threshold
        or s["range_ratio"] > args.threshold * 2
        or s["ch_absmax_ratio"] > args.threshold
    )
    print(f"\n  Flagged {n_flagged}/{len(all_stats)} tensors as potentially problematic.")
    print(f"  The 'ch_range_ratio' metric measures how much the dynamic range")
    print(f"  varies across channels — ratios > {args.threshold} indicate uneven")
    print(f"  activation distributions that lose precision under per-tensor quantisation.\n")


if __name__ == "__main__":
    main()
