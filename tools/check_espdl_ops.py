#!/usr/bin/env python3
"""
check_espdl_ops.py — ESP-DL operator compatibility checker for ONNX models.

Checks every op in one or more ONNX graphs against the ESP-DL support matrix
(operator_support_state.md, opset 18, generated 2026-03-06).

Usage:
    python check_espdl_ops.py model.onnx [model2.onnx ...]
    python check_espdl_ops.py ~/Models/*.onnx --target esp32s3
    python check_espdl_ops.py model.onnx --show-all --json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# ESP-DL support matrix
# Source: operator_support_state.md (generated 2026-03-06)
# Keys are ONNX op names. Values:
#   int8    : bool
#   int16   : bool
#   float32 : bool
#   note    : str  (restriction summary, empty string if none)
# ---------------------------------------------------------------------------
ESP_DL_OPS: dict[str, dict] = {
    "Add":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "AveragePool":        {"int8": True,  "int16": True,  "float32": True,  "note": "1d/2d; no dilation"},
    "Clip":               {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Concat":             {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Conv":               {"int8": True,  "int16": True,  "float32": False, "note": "1d/2d; groups=1 or depthwise only; NO float32"},
    "ConvTranspose":      {"int8": True,  "int16": True,  "float32": False, "note": "Via InsertZeros+Conv; NO float32"},
    "DepthToSpace":       {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Div":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "Elu":                {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Equal":              {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "Exp":                {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Flatten":            {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Gather":             {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Gemm":               {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "GlobalAveragePool":  {"int8": True,  "int16": True,  "float32": True,  "note": "1d/2d"},
    "Greater":            {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "GreaterOrEqual":     {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "GRU":                {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "HardSigmoid":        {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "HardSwish":          {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "LayerNormalization": {"int8": True,  "int16": True,  "float32": True,  "note": "With scale and bias"},
    "LeakyRelu":          {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Less":               {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "LessOrEqual":        {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "Log":                {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "LogSoftmax":         {"int8": True,  "int16": True,  "float32": True,  "note": "Output dtype is float32"},
    "LSTM":               {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "MatMul":             {"int8": True,  "int16": True,  "float32": False, "note": "Up to 4D; NO float32"},
    "MaxPool":            {"int8": True,  "int16": True,  "float32": False, "note": "1d/2d; no dilation; NO float32"},
    "Mod":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D; fmod=1 only"},
    "Mul":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "Neg":                {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Pad":                {"int8": True,  "int16": True,  "float32": True,  "note": "No wrap mode"},
    "Pow":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D; multidirectional broadcast"},
    "PRelu":              {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "ReduceL1":           {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceL2":           {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceLogSum":       {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceLogSumExp":    {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceMax":          {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceMean":         {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceMin":          {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceProd":         {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceSum":          {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "ReduceSumSquare":    {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "Relu":               {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Reshape":            {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Resize":             {"int8": True,  "int16": False, "float32": False, "note": "int8 only; 1d/2d nearest/linear; NO float32"},
    "ReverseSequence":    {"int8": True,  "int16": True,  "float32": False, "note": "NO float32"},
    "ScatterND":          {"int8": True,  "int16": True,  "float32": True,  "note": "Reductions: none/add/mul/max/min"},
    "Sigmoid":            {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Slice":              {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Softmax":            {"int8": True,  "int16": True,  "float32": True,  "note": "Output dtype is float32"},
    "SpaceToDepth":       {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Split":              {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Sqrt":               {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Squeeze":            {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Sub":                {"int8": True,  "int16": True,  "float32": True,  "note": "Up to 4D"},
    "Swish":              {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Tanh":               {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Transpose":          {"int8": True,  "int16": True,  "float32": True,  "note": ""},
    "Unsqueeze":          {"int8": True,  "int16": True,  "float32": True,  "note": ""},
}

# ---------------------------------------------------------------------------
# Mamba-specific ops that are never in any standard op library.
# We annotate them specially so the report is self-explanatory.
# ---------------------------------------------------------------------------
MAMBA_CUSTOM_OPS = {
    "SelectiveScan", "MambaSelectiveScan", "SSMKernel",
    "CumSum",  # used in some SSM export paths
}

ANSI_RED    = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_GREEN  = "\033[92m"
ANSI_CYAN   = "\033[96m"
ANSI_BOLD   = "\033[1m"
ANSI_RESET  = "\033[0m"

def supports_color() -> bool:
    return sys.stdout.isatty()

def c(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}" if supports_color() else text


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------
def load_onnx(path: Path):
    try:
        import onnx
    except ImportError:
        sys.exit("ERROR: onnx not installed. Run: pip install onnx --break-system-packages")
    model = onnx.load(str(path))
    return model


def collect_ops(model) -> dict[str, list[str]]:
    """
    Walk the main graph (and any sub-graphs in If/Loop/Scan nodes) and
    return a dict mapping op_type -> [node_name, ...].
    """
    op_to_nodes: dict[str, list[str]] = defaultdict(list)

    def walk_graph(graph):
        for node in graph.node:
            label = node.name or f"<anon:{node.op_type}>"
            op_to_nodes[node.op_type].append(label)
            # Recurse into sub-graphs (If, Loop, Scan body attributes)
            for attr in node.attribute:
                import onnx
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        walk_graph(g)

    walk_graph(model.graph)
    return dict(op_to_nodes)


def get_opset(model) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return -1


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
TARGET_DTYPES = ("int8", "int16", "float32")

def analyse(op_to_nodes: dict[str, list[str]], target: str) -> dict:
    """
    Returns structured result dict with supported / float32_only /
    unsupported / unknown categories.
    """
    # Determine which dtype column to check for the target platform
    target_dtype_flags = {
        "esp32":   ("int8",),
        "esp32s3": ("int8", "int16"),
        "esp32p4": ("int8", "int16"),
    }.get(target, ("int8", "int16"))

    supported        = {}  # op -> {count, note, f32_restricted}
    unsupported      = {}  # op -> {count, reason}

    for op, nodes in sorted(op_to_nodes.items()):
        count = len(nodes)
        if op in MAMBA_CUSTOM_OPS:
            unsupported[op] = {
                "count": count,
                "reason": "Mamba-specific custom op — no standard runtime support",
                "nodes": nodes,
            }
        elif op in ESP_DL_OPS:
            info = ESP_DL_OPS[op]
            # Is it usable at all on the target (ignoring dtype)?
            platform_ok = any(info[d] for d in target_dtype_flags)
            f32_restricted = not info["float32"]
            if platform_ok:
                supported[op] = {
                    "count": count,
                    "note": info["note"],
                    "float32_ok": info["float32"],
                    "nodes": nodes,
                }
            else:
                unsupported[op] = {
                    "count": count,
                    "reason": f"Not supported on {target} at any dtype",
                    "nodes": nodes,
                }
        else:
            unsupported[op] = {
                "count": count,
                "reason": "Not in ESP-DL operator list",
                "nodes": nodes,
            }

    f32_warnings = {op: v for op, v in supported.items() if not v["float32_ok"]}

    return {
        "supported": supported,
        "unsupported": unsupported,
        "f32_warnings": f32_warnings,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def print_report(path: Path, opset: int, result: dict, target: str, show_all: bool) -> None:
    supported   = result["supported"]
    unsupported = result["unsupported"]
    f32_warn    = result["f32_warnings"]

    total = len(supported) + len(unsupported)
    n_ok  = len(supported)
    n_bad = len(unsupported)
    n_f32 = len(f32_warn)

    print()
    print(c("=" * 72, ANSI_BOLD))
    print(c(f"  {path.name}", ANSI_BOLD))
    print(c("=" * 72, ANSI_BOLD))
    print(f"  Opset        : {opset}  (ESP-DL recommends 18)")
    print(f"  Target       : {target.upper()}")
    print(f"  Unique ops   : {total}  "
          f"({c(str(n_ok) + ' supported', ANSI_GREEN)}, "
          f"{c(str(n_bad) + ' unsupported', ANSI_RED)}"
          + (f", {c(str(n_f32) + ' float32-restricted', ANSI_YELLOW)}" if n_f32 else "")
          + ")")
    print()

    # ---- Unsupported ops (show always) ------------------------------------
    if unsupported:
        print(c("  ✗ UNSUPPORTED OPS", ANSI_RED + ANSI_BOLD))
        print(c("  " + "-" * 68, ANSI_RED))
        col_w = max(len(op) for op in unsupported) + 2
        for op, v in sorted(unsupported.items()):
            tag  = c(op.ljust(col_w), ANSI_RED)
            cnt  = f"×{v['count']}"
            reason = v["reason"]
            print(f"    {tag} {cnt:>4}   {reason}")
        print()
    else:
        print(c("  ✓ All ops are in the ESP-DL operator set.\n", ANSI_GREEN))

    # ---- float32 warnings (show always) ------------------------------------
    if f32_warn:
        print(c("  ⚠  FLOAT32-RESTRICTED OPS  (int8/int16 quantization required)", ANSI_YELLOW + ANSI_BOLD))
        print(c("  " + "-" * 68, ANSI_YELLOW))
        col_w = max(len(op) for op in f32_warn) + 2
        for op, v in sorted(f32_warn.items()):
            tag = c(op.ljust(col_w), ANSI_YELLOW)
            cnt = f"×{v['count']}"
            note = v["note"] if v["note"] else ""
            print(f"    {tag} {cnt:>4}   {note}")
        print()

    # ---- Supported ops (optional) -----------------------------------------
    if show_all and supported:
        print(c("  ✓ SUPPORTED OPS", ANSI_GREEN + ANSI_BOLD))
        print(c("  " + "-" * 68, ANSI_GREEN))
        col_w = max(len(op) for op in supported) + 2
        for op, v in sorted(supported.items()):
            tag  = c(op.ljust(col_w), ANSI_GREEN)
            cnt  = f"×{v['count']}"
            note = (" [" + v["note"] + "]") if v["note"] else ""
            f32  = "" if v["float32_ok"] else c(" [no float32]", ANSI_YELLOW)
            print(f"    {tag} {cnt:>4}{note}{f32}")
        print()


def print_summary(results: list[tuple[Path, dict]]) -> None:
    if len(results) < 2:
        return
    print(c("=" * 72, ANSI_BOLD))
    print(c("  SUMMARY ACROSS ALL MODELS", ANSI_BOLD))
    print(c("=" * 72, ANSI_BOLD))
    all_unsupported: dict[str, set[str]] = defaultdict(set)
    all_f32warn:     dict[str, set[str]] = defaultdict(set)
    for path, result in results:
        for op in result["unsupported"]:
            all_unsupported[op].add(path.name)
        for op in result["f32_warnings"]:
            all_f32warn[op].add(path.name)

    if all_unsupported:
        print(c("\n  Unsupported ops across all models:", ANSI_RED))
        for op, models in sorted(all_unsupported.items()):
            print(f"    {c(op, ANSI_RED)} — in: {', '.join(sorted(models))}")
    else:
        print(c("\n  All ops supported across all models.", ANSI_GREEN))

    if all_f32warn:
        print(c("\n  Float32-restricted ops across all models:", ANSI_YELLOW))
        for op, models in sorted(all_f32warn.items()):
            print(f"    {c(op, ANSI_YELLOW)} — in: {', '.join(sorted(models))}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Check ONNX model ops against the ESP-DL support matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("models", nargs="+", type=Path, help="ONNX model file(s)")
    p.add_argument(
        "--target", default="esp32s3",
        choices=["esp32", "esp32s3", "esp32p4"],
        help="Target ESP32 variant (default: esp32s3)",
    )
    p.add_argument(
        "--show-all", action="store_true",
        help="Also print the list of supported ops (default: unsupported + warnings only)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Dump raw results as JSON to stdout",
    )
    return p.parse_args()


def main():
    args = parse_args()

    all_results = []
    json_out = {}

    for path in args.models:
        if not path.exists():
            print(c(f"ERROR: file not found: {path}", ANSI_RED), file=sys.stderr)
            continue

        model  = load_onnx(path)
        opset  = get_opset(model)
        ops    = collect_ops(model)
        result = analyse(ops, args.target)

        all_results.append((path, result))

        if not args.json:
            print_report(path, opset, result, args.target, args.show_all)
        else:
            json_out[str(path)] = {
                "opset": opset,
                "target": args.target,
                "supported":   {op: {"count": v["count"], "note": v["note"], "float32_ok": v["float32_ok"]}
                                 for op, v in result["supported"].items()},
                "unsupported": {op: {"count": v["count"], "reason": v["reason"]}
                                 for op, v in result["unsupported"].items()},
                "f32_warnings":{op: {"count": v["count"], "note": v["note"]}
                                 for op, v in result["f32_warnings"].items()},
            }

    if args.json:
        print(json.dumps(json_out, indent=2))
    else:
        print_summary(all_results)
        # Exit code: 0 = all good, 1 = unsupported ops found
        any_unsupported = any(r["unsupported"] for _, r in all_results)
        sys.exit(1 if any_unsupported else 0)


if __name__ == "__main__":
    main()
