#!/usr/bin/env python3
"""
simplify_for_espdl.py — ONNX graph simplification + Softplus decomposition
for ESP-DL deployment.

Pass 1 : onnxsim  — folds Shape/Constant/ConstantOfShape/Cast/Expand/Where
         with static input shapes.
Pass 2 : Graph surgery — replaces Softplus with Log(Add(Exp(x), 1)),
         using only ops available in ESP-DL.

Usage:
    python simplify_for_espdl.py src/models/har-mamba-1.onnx
    python simplify_for_espdl.py src/models/*.onnx --suffix _simplified
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


# ---------------------------------------------------------------------------
# Pass 1: onnxsim with static shapes
# ---------------------------------------------------------------------------
# Input shapes for each dataset — update if you add models.
KNOWN_SHAPES: dict[str, dict[str, list[int]]] = {
    "har": {"input": [1, 10, 57]},
    "kws": {"input": [1, 101, 40]},
}

def detect_dataset(path: Path) -> str | None:
    """Heuristic: infer dataset from filename."""
    name = path.stem.lower()
    if "har" in name:
        return "har"
    if "kws" in name or "speech" in name or "keyword" in name:
        return "kws"
    return None


def find_input_name(model: onnx.ModelProto) -> str:
    """Return the name of the first graph input."""
    return model.graph.input[0].name


def simplify(model: onnx.ModelProto, input_shapes: dict[str, list[int]]) -> onnx.ModelProto:
    try:
        from onnxsim import simplify as onnxsim
    except ImportError:
        sys.exit("ERROR: onnxsim not installed. Run: pip install onnxsim --break-system-packages")

    simplified, ok = onnxsim(model, input_shapes=input_shapes)
    if not ok:
        print("  WARNING: onnxsim reported partial simplification — proceeding anyway.")
    return simplified


# ---------------------------------------------------------------------------
# Pass 2: Softplus decomposition
# Softplus(x) = Log(Add(Exp(x), Constant(1.0)))
# ---------------------------------------------------------------------------
def decompose_softplus(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Replace every Softplus node with Exp → Add → Log.
    Returns (new_model, count_replaced).
    """
    graph = model.graph
    new_nodes = []
    replaced = 0

    for node in graph.node:
        if node.op_type != "Softplus":
            new_nodes.append(node)
            continue

        assert len(node.input) == 1 and len(node.output) == 1, \
            f"Unexpected Softplus shape: inputs={node.input}, outputs={node.output}"

        x_name   = node.input[0]
        out_name = node.output[0]
        prefix   = out_name  # use output name as prefix to guarantee uniqueness

        exp_out  = f"{prefix}__softplus_exp"
        one_name = f"{prefix}__softplus_one"
        add_out  = f"{prefix}__softplus_add"

        # Add 1.0 as a graph initializer (a stored weight), NOT as a Constant
        # op node — initializers are invisible to op-compatibility checkers and
        # are the correct ONNX representation for fixed scalar parameters.
        one_initializer = numpy_helper.from_array(
            np.array([1.0], dtype=np.float32),
            name=one_name,
        )
        graph.initializer.append(one_initializer)

        exp_node = helper.make_node(
            "Exp", inputs=[x_name], outputs=[exp_out],
            name=f"{prefix}__softplus_exp_node",
        )
        add_node = helper.make_node(
            "Add", inputs=[exp_out, one_name], outputs=[add_out],
            name=f"{prefix}__softplus_add_node",
        )
        log_node = helper.make_node(
            "Log", inputs=[add_out], outputs=[out_name],
            name=f"{prefix}__softplus_log_node",
        )

        new_nodes.extend([exp_node, add_node, log_node])
        replaced += 1
        print(f"    Decomposed Softplus: {node.name or '<anon>'} → Exp+Add+Log")

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Re-run shape inference after surgery
    model = onnx.shape_inference.infer_shapes(model)
    return model, replaced


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def count_ops(model: onnx.ModelProto) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(n.op_type for n in model.graph.node))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(path: Path, suffix: str, input_name_override: str | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    model = onnx.load(str(path))
    before = count_ops(model)
    print(f"  Ops before  : {sum(before.values())} nodes, {len(before)} unique types")

    dataset = detect_dataset(path)
    if dataset is None:
        print("  WARNING: could not detect dataset from filename.")
        print("           Pass --input-shape or edit KNOWN_SHAPES in this script.")
        input_shapes = {}
    else:
        input_node_name = input_name_override or find_input_name(model)
        raw_shape = KNOWN_SHAPES[dataset]
        input_shapes = {input_node_name: raw_shape["input"]}
        print(f"  Dataset     : {dataset.upper()}  →  input '{input_node_name}' = {raw_shape['input']}")

    # Pass 1: onnxsim
    print("\n  [Pass 1] onnxsim constant folding …")
    model = simplify(model, input_shapes)
    after_sim = count_ops(model)
    folded = sum(before.values()) - sum(after_sim.values())
    print(f"    Folded {folded} nodes. Remaining: {sum(after_sim.values())} nodes, {len(after_sim)} unique types")

    # Pass 2: Softplus decomposition
    print("\n  [Pass 2] Softplus decomposition …")
    model, n_softplus = decompose_softplus(model)
    if n_softplus == 0:
        print("    No Softplus nodes found (already folded or not present).")

    # Final op count
    after = count_ops(model)
    print(f"\n  Ops after   : {sum(after.values())} nodes, {len(after)} unique types")

    # Diff summary
    new_types  = set(after) - set(before)
    gone_types = set(before) - set(after)
    if gone_types:
        print(f"  Removed op types : {', '.join(sorted(gone_types))}")
    if new_types:
        print(f"  New op types     : {', '.join(sorted(new_types))}  (from Softplus decomposition)")

    # Save
    out_path = path.with_stem(path.stem + suffix)
    onnx.save(model, str(out_path))
    print(f"\n  Saved → {out_path}")
    print(f"\n  Run check_espdl_ops.py on the simplified model to verify.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument(
        "--suffix", default="_simplified",
        help="Suffix appended to output filenames (default: _simplified)",
    )
    p.add_argument(
        "--input-name", default=None,
        help="Override the graph input node name (auto-detected by default)",
    )
    args = p.parse_args()

    for path in args.models:
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            continue
        process(path, args.suffix, args.input_name)


if __name__ == "__main__":
    main()
