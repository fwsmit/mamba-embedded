#!/usr/bin/env python3
"""
count_mamba_macs.py
--------------------
Compute MACs / FLOPs / params for an ONNX Mamba model, with a breakdown
that separates:

  1. "Linear algebra" MACs      -> MatMul, Conv, Gemm
  2. "SSM scan" MACs            -> Mul + ReduceSum nodes inside the
                                    selective-scan recurrence (ONNX has no
                                    native fused "scan" op, so exported
                                    Mamba graphs decompose h_t = A_bar*h_{t-1}
                                    + B_bar*x_t into elementwise Mul/Add/
                                    ReduceSum nodes, one unrolled step per
                                    timestep)
  3. "Nonlinear/norm" op cost   -> Sigmoid, Exp, Log, Softplus, Relu, Neg,
                                    LayerNorm, ReduceMean, etc.

(1)+(2) is the strict "MACs" figure (matches how most CNN papers define
MACs: weight-bearing linear-algebra ops only).

(3) is NOT a multiply-accumulate -- it's onnx-tool's *instruction-cost*
heuristic for transcendental ops, based on how many vectorized x86
instructions a polynomial approximation of that function takes (see
onnx_tool/node.py, citing https://github.com/reyoung/avx_mathfun):

    MUL=1, ADD=1, CMP=1, DIV=4 (vrcp14ps), EXP=32, LOG=43,
    TANH = EXP + 2*ADD + DIV = 38, Sigmoid costed as EXP (32/elem)

For a Mamba block this bucket is NOT negligible: the exp()-based
discretization (A_bar = exp(dt * A)) and the SiLU/sigmoid gates make
the SSM meaningfully more activation-heavy than a CNN, so reporting
only (1)+(2) can understate real compute cost. This script therefore
reports THREE numbers:

    - "MACs (strict)"            = linear + scan MACs only
    - "Nonlinear op cost"        = heuristic instruction-equivalent cost
    - "Total compute cost"       = MACs (strict) + nonlinear op cost,
                                    i.e. the full instruction-equivalent
                                    complexity of the model

Report the strict MACs figure when comparing against papers that use a
standard MAC/FLOP counter (they will not have counted transcendental ops
this way). Report the "total compute cost" figure alongside it, clearly
labeled with the instruction-cost convention above, when you want a
number that reflects the real activation-function burden of the SSM --
this is the more defensible choice if a reviewer asks why your model's
measured on-device latency doesn't track its MAC count.

Usage
-----
    pip install onnx onnx-tool --break-system-packages

    python count_mamba_macs.py model.onnx
    python count_mamba_macs.py model.onnx --csv breakdown.csv
    python count_mamba_macs.py model.onnx --input-shape input:1,10,57
    python count_mamba_macs.py model_a.onnx model_b.onnx   # compare several

Notes on dynamic shapes
------------------------
If your ONNX graph has dynamic axes (batch size, sequence length), you
must supply concrete shapes via --input-shape NAME:d0,d1,d2 (repeatable)
or the shape inferencer will fail / silently skip nodes. Use the exact
input tensor name(s) as they appear in the model (the script will print
them if you get it wrong).
"""

import argparse
import csv
import sys
from collections import defaultdict

import onnx
import onnx_tool

# Ops that constitute genuine multiply-accumulate operations in standard
# linear algebra (weight-bearing layers).
LINEAR_OPS = {"MatMul", "Conv", "Gemm", "ConvTranspose"}

# Ops that, in an exported/unrolled Mamba selective-scan, implement the
# per-timestep recurrence h_t = A_bar * h_{t-1} + B_bar * x_t. Each
# Mul contributes the "multiply" half and each ReduceSum the "accumulate"
# half of a MAC, so counting elementwise output size for both is the
# correct convention (equivalent to a matmul decomposed by hand).
SCAN_OPS = {"Mul", "ReduceSum"}

# Ops with real compute cost that is NOT a multiply-accumulate -- these
# get a heuristic "cost" from onnx-tool but should be reported separately,
# if at all, since the FLOP-equivalence convention varies by tool/paper.
NONLINEAR_OPS = {
    "Sigmoid", "Exp", "Log", "Softplus", "Relu", "Neg", "Tanh", "Erf",
    "LayerNormalization", "ReduceMean", "Softmax", "Pow", "Sqrt", "Div",
}

# Everything else (Reshape, Transpose, Gather, Slice, Concat, Unsqueeze,
# Split, ...) is treated as free (0 MACs / negligible cost), which is the
# standard convention -- these are memory-layout ops, not compute.


def _as_int(x):
    """onnx-tool sometimes returns macs/params as a list (per-output); sum it."""
    if isinstance(x, (list, tuple)):
        return int(sum(x))
    return int(x)


def parse_shape_overrides(shape_args):
    """Parse --input-shape NAME:d0,d1,d2 entries into {name: [d0,d1,d2]}."""
    overrides = {}
    for entry in shape_args or []:
        if ":" not in entry:
            raise ValueError(
                f"--input-shape must be NAME:d0,d1,... got '{entry}'"
            )
        name, dims = entry.split(":", 1)
        overrides[name] = [int(d) for d in dims.split(",")]
    return overrides


def apply_shape_overrides(model_path, overrides):
    """Return an onnx ModelProto with fixed input shapes, or the original
    path if no overrides were given / no dynamic dims are present."""
    if not overrides:
        return model_path

    model = onnx.load(model_path)
    for inp in model.graph.input:
        if inp.name not in overrides:
            continue
        dims = overrides[inp.name]
        tensor_type = inp.type.tensor_type
        if len(dims) != len(tensor_type.shape.dim):
            raise ValueError(
                f"Input '{inp.name}' has {len(tensor_type.shape.dim)} dims "
                f"in the model but you passed {len(dims)}."
            )
        for d, val in zip(tensor_type.shape.dim, dims):
            d.Clear()
            d.dim_value = val

    fixed_path = model_path.replace(".onnx", "") + "_fixed_shapes.onnx"
    onnx.save(model, fixed_path)
    return fixed_path


def profile_model(model_path):
    """Run onnx-tool's shape inference + profiler and return a per-node list."""
    m = onnx_tool.Model(model_path)
    m.graph.shape_infer()
    m.graph.profile()

    rows = []
    for name, node in m.graph.nodemap.items():
        rows.append({
            "name": name,
            "op_type": node.op_type,
            "macs": _as_int(getattr(node, "macs", 0)),
            "params": _as_int(getattr(node, "params", 0)),
            "in_shape": getattr(node, "inshape", None),
            "out_shape": getattr(node, "outshape", None),
        })
    return rows


def summarize(rows):
    by_op_macs = defaultdict(int)
    by_op_count = defaultdict(int)
    for r in rows:
        by_op_macs[r["op_type"]] += r["macs"]
        by_op_count[r["op_type"]] += 1

    linear_macs = sum(v for k, v in by_op_macs.items() if k in LINEAR_OPS)
    scan_macs = sum(v for k, v in by_op_macs.items() if k in SCAN_OPS)
    nonlinear_cost = sum(v for k, v in by_op_macs.items() if k in NONLINEAR_OPS)
    other_macs = sum(
        v for k, v in by_op_macs.items()
        if k not in LINEAR_OPS and k not in SCAN_OPS and k not in NONLINEAR_OPS
    )

    true_macs = linear_macs + scan_macs  # <- strict "MACs" figure
    total_compute_cost = true_macs + nonlinear_cost  # <- MACs + heuristic op cost

    return {
        "by_op_macs": dict(by_op_macs),
        "by_op_count": dict(by_op_count),
        "linear_macs": linear_macs,
        "scan_macs": scan_macs,
        "nonlinear_cost": nonlinear_cost,
        "other_macs": other_macs,
        "true_macs": true_macs,
        "total_compute_cost": total_compute_cost,
    }


def count_params(model_path):
    """Sum sizes of all initializer tensors -- robust ground truth for
    parameter count, independent of onnx-tool's node-level accounting
    (which can double count weights shared across nodes)."""
    model = onnx.load(model_path)
    total = 0
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        total += arr.size
    return total


def print_report(model_path, rows, summary, params):
    print("=" * 72)
    print(f"Model: {model_path}")
    print("=" * 72)

    print(f"{'Op type':22s}{'count':>8s}{'MACs':>14s}")
    print("-" * 44)
    for op, macs in sorted(summary["by_op_macs"].items(), key=lambda kv: -kv[1]):
        tag = ""
        if op in LINEAR_OPS:
            tag = "  [linear]"
        elif op in SCAN_OPS:
            tag = "  [scan]"
        elif op in NONLINEAR_OPS:
            tag = "  [nonlinear, not a MAC]"
        print(f"{op:22s}{summary['by_op_count'][op]:>8d}{macs:>14,d}{tag}")

    print("-" * 44)
    print()
    print(f"  Linear-algebra MACs (MatMul/Conv/Gemm)   : {summary['linear_macs']:>12,d}")
    print(f"  SSM selective-scan MACs (Mul/ReduceSum)  : {summary['scan_macs']:>12,d}")
    if summary["other_macs"]:
        print(f"  Other uncategorized MACs                 : {summary['other_macs']:>12,d}")
    print(f"  {'-'*56}")
    print(f"  MACs (strict, linear+scan only)          : {summary['true_macs']:>12,d}")
    print(f"  Approx. FLOPs (2x strict MACs)            : {2*summary['true_macs']:>12,d}")
    print()
    print(f"  Nonlinear/norm op cost (instruction-eq.) : {summary['nonlinear_cost']:>12,d}"
          f"   [EXP=32,LOG=43,DIV=4,CMP=1 per elem; see avx_mathfun]")
    print(f"  {'-'*56}")
    print(f"  TOTAL COMPUTE COST (MACs + nonlinear)    : {summary['total_compute_cost']:>12,d}")
    print()
    print(f"  Parameters (from initializers)            : {params:>12,d}")
    print()
    print("  Reporting guidance:")
    print("    - Use 'MACs (strict)' when comparing to papers using a")
    print("      standard MAC/FLOP counter (thop/fvcore/ptflops).")
    print("    - Use 'TOTAL COMPUTE COST' when you want a number that")
    print("      reflects the SSM's real activation-function burden --")
    print("      cite the instruction-cost convention (avx_mathfun-based")
    print("      polynomial approximation costs) if you report it.")
    print()


def write_csv(rows, summary, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_name", "op_type", "macs", "params", "category"])
        for r in rows:
            if r["op_type"] in LINEAR_OPS:
                cat = "linear"
            elif r["op_type"] in SCAN_OPS:
                cat = "scan"
            elif r["op_type"] in NONLINEAR_OPS:
                cat = "nonlinear (not a MAC)"
            else:
                cat = "other"
            writer.writerow([r["name"], r["op_type"], r["macs"], r["params"], cat])
        writer.writerow([])
        writer.writerow(["TOTAL_linear_macs", "", summary["linear_macs"]])
        writer.writerow(["TOTAL_scan_macs", "", summary["scan_macs"]])
        writer.writerow(["TOTAL_macs_strict (linear+scan)", "", summary["true_macs"]])
        writer.writerow(["TOTAL_nonlinear_op_cost (instruction-equivalent)", "", summary["nonlinear_cost"]])
        writer.writerow(["TOTAL_compute_cost (macs_strict + nonlinear_op_cost)", "", summary["total_compute_cost"]])
    print(f"Per-node breakdown written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Count MACs/FLOPs/params for ONNX Mamba models, "
                    "separating linear-algebra MACs, SSM-scan MACs, and "
                    "nonlinear-op heuristic cost."
    )
    parser.add_argument("models", nargs="+", help="Path(s) to .onnx file(s)")
    parser.add_argument(
        "--input-shape", action="append", default=[],
        help="Fix a dynamic input shape: NAME:d0,d1,d2 (repeatable for "
             "multiple inputs). Required if the graph has dynamic axes."
    )
    parser.add_argument(
        "--csv", default=None,
        help="Write a per-node CSV breakdown (only valid with one model)."
    )
    args = parser.parse_args()

    if args.csv and len(args.models) > 1:
        print("--csv only supported for a single model at a time.", file=sys.stderr)
        sys.exit(1)

    overrides = parse_shape_overrides(args.input_shape)

    for model_path in args.models:
        try:
            resolved_path = apply_shape_overrides(model_path, overrides)
        except ValueError as e:
            print(f"[{model_path}] shape override error: {e}", file=sys.stderr)
            # Show input names/shapes to help the user fix their command
            m = onnx.load(model_path)
            print("  Model inputs:")
            for inp in m.graph.input:
                dims = [
                    d.dim_value if d.dim_value else (d.dim_param or "?")
                    for d in inp.type.tensor_type.shape.dim
                ]
                print(f"    {inp.name}: {dims}")
            sys.exit(1)

        try:
            rows = profile_model(resolved_path)
        except Exception as e:
            print(f"[{model_path}] failed to profile: {e}", file=sys.stderr)
            print(
                "  This usually means the graph has dynamic shapes that "
                "need --input-shape, or an op onnx-tool doesn't recognize.",
                file=sys.stderr,
            )
            continue

        summary = summarize(rows)
        params = count_params(resolved_path)
        print_report(model_path, rows, summary, params)

        if args.csv:
            write_csv(rows, summary, args.csv)


if __name__ == "__main__":
    main()
