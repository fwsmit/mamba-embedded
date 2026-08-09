#!/usr/bin/env python3
"""
quantize_kws_espdl.py
======================

End-to-end 8-bit (mixed 8/16-bit) quantization of the Mamba KWS model for
ESP32-S3, using esp-ppq.

WHAT THIS SCRIPT DOES
----------------------
1. Patches a small ONNX export quirk (see PATCH_KEEPDIMS below) that
   otherwise crashes the esp-ppq exporter.
2. Calibrates and quantizes the model to INT8, EXCEPT for the operators
   that implement the Mamba selective-scan recurrence, which are quantized
   to INT16 instead. This is the change that matters (see REPORT.md for
   the full explanation) -- everything else is default esp-ppq PTQ.
3. Exports a .espdl file for deployment on the ESP32-S3.
4. Evaluates float (ONNX Runtime) and quantized (esp-ppq TorchExecutor)
   accuracy on a held-out test set so you can see the effect directly.

USAGE
-----
    python quantize_kws_espdl.py \
        --onnx mamba-1-kws-2-test.onnx \
        --val val.pkl --test test.pkl \
        --output mamba-1-kws-2-test-int8.espdl

Both val.pkl and test.pkl are expected to be dicts with keys "X" (N,49,40
float32 MFCCs) and "y" (N, int64 labels), as produced by the original
training pipeline's SpeechCommandsMFCC.save().

Requires: esp-ppq, torch, onnx, onnxruntime, numpy
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from torch.utils.data import DataLoader, Dataset, Subset

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import TargetPlatform
from esp_ppq.executor import TorchExecutor
from esp_ppq.quantization.optim.training import TrainingBasedPass
from esp_ppq.quantization.algorithm.training import BlockBuilder

INPUT_SHAPE = [1, 49, 40]

# --------------------------------------------------------------------------
# Block-construction instrumentation
# --------------------------------------------------------------------------
# TrainingBasedPass.split_graph_into_blocks() is what turns `block_size`
# into an actual list of TrainableBlocks (see esp_ppq/quantization/optim/
# training.py). TrainedQuantizationThresholdPass.optimize() already prints
# `sp.name -> ep.name` per block as it fine-tunes, but not how many ops
# each block actually contains -- which is the number we need to confirm
# blocks are growing to span the recurrence (or not) as block_size grows.
# We monkeypatch the method itself (not the installed package files) so
# this is purely a logging hook: behavior is unchanged, we just print
# block sp/ep/op-count right after construction, before any training
# happens.
_orig_split_graph_into_blocks = TrainingBasedPass.split_graph_into_blocks

# When set (list of ONNX node names, in order), split_graph_into_blocks is
# bypassed entirely and instead builds N-1 blocks tiling the WHOLE graph
# end-to-end, with boundaries at these exact ops -- e.g. a run of evenly
# spaced ReduceSum ops inside the SSM scan (one ReduceSum per unrolled
# timestep in this graph). This exists because the *default* algorithm can
# only ever start a block at a computing op (Conv/MatMul/Gemm), so there is
# no `block_size` value that produces several blocks whose boundaries sit
# *inside* the scan -- BlockBuilder.create_block(sp, ep) has no such
# restriction (it accepts any two ops), so we call it directly.
_CUSTOM_BLOCK_BOUNDARY_NAMES = None


def _instrumented_split_graph_into_blocks(self, graph, executing_order, blocksize=None,
                                           overlap=False, interested_layers=None):
    if _CUSTOM_BLOCK_BOUNDARY_NAMES is not None:
        pairs = _CUSTOM_BLOCK_BOUNDARY_NAMES  # list of (sp_name, ep_name) tuples
        builder = BlockBuilder(graph=graph, topo_order=executing_order)
        blocks = []
        for a, b in pairs:
            sp, ep = graph.operations[a], graph.operations[b]
            blocks.append(builder.create_block(sp=sp, ep=ep))
        print(f"[block-log] CUSTOM boundaries -> {len(blocks)} block(s) "
              f"(bypassing is_computing_op anchoring)")
    else:
        blocks = _orig_split_graph_into_blocks(
            self, graph, executing_order, blocksize=blocksize,
            overlap=overlap, interested_layers=interested_layers,
        )
        print(f"[block-log] split_graph_into_blocks(block_size={blocksize}) "
              f"-> {len(blocks)} block(s)")
    for i, block in enumerate(blocks):
        print(f"[block-log]   Block[{i + 1}/{len(blocks)}]: "
              f"{block.sp.name} ({block.sp.type}) -> "
              f"{block.ep.name} ({block.ep.type})  n_ops={len(block.rps)}")
    return blocks


TrainingBasedPass.split_graph_into_blocks = _instrumented_split_graph_into_blocks


_CUSTOM_BLOCK_MODE = "prefix"  # 'prefix' (only valid mode, see below) or 'segments' (invalid, kept for the record)


def set_custom_scan_blocks(n_blocks: int, onnx_path: Path):
    """Build `n_blocks` TrainableBlocks that give intermediate supervision
    at increasing depths through the SSM scan.

    IMPORTANT: a naive disjoint-segment partition (block k = [ReduceSum_i ->
    ReduceSum_j], one block per ~7-timestep chunk) is NOT valid here and
    fails at runtime. TrainableBlock requires 'S must be on every path from
    graph input to E' (see BlockBuilder.build docstring) -- but a mid-scan
    op like ReduceSum_7 is *not* actually on every path to ReduceSum_14:
    each timestep's ops also read directly from tensors computed once,
    upstream, near dt_proj (per-timestep Gather-indexed slices), not
    solely from the previous timestep's carried state. Confirmed
    empirically: block 1 (linear_in -> ReduceSum_7) trains fine; block 2
    (ReduceSum_7 -> ReduceSum_14) crashes inside partial_graph_forward
    with "ReduceSum_14 ... is not in list" because the executor can't
    resolve that extra, non-block-local input dependency.

    The valid alternative used here: PREFIX blocks. Every block starts at
    the graph's true first op (trivially on every path to anything) and
    ends at a progressively deeper checkpoint (ReduceSum_7, _14, ..., and
    finally the graph's true last op). This is the closest architecturally
    valid approximation to "blocks spread across the scan" -- each block's
    reconstruction loss is evaluated at a *different depth* through the
    unrolled recurrence (t=7, 14, 21, ..., 49) instead of only once at the
    very end, at the cost of blocks now being nested/overlapping (block k
    contains all of block k-1) rather than disjoint, so training is no
    longer "one pass over the whole graph" but n_blocks sequential passes
    each re-touching the shared upstream ops.
    """
    global _CUSTOM_BLOCK_BOUNDARY_NAMES
    m = onnx.load(str(onnx_path))
    nodes = list(m.graph.node)
    reducesum_names = [n.name for n in nodes if n.op_type == "ReduceSum"]
    n_steps = len(reducesum_names)  # 49 unrolled timesteps in this model
    step_boundaries = [round(i * n_steps / n_blocks) for i in range(1, n_blocks)]
    checkpoints = [reducesum_names[i] for i in step_boundaries] + [nodes[-1].name]
    first_name = nodes[0].name
    _CUSTOM_BLOCK_BOUNDARY_NAMES = [(first_name, cp) for cp in checkpoints]
    print(f"[quantize] custom PREFIX scan blocks: {n_blocks} blocks, "
          f"each [{first_name} -> checkpoint], checkpoints={checkpoints}")

# Elementwise op types that make up the unrolled SSM scan (no weights of
# their own -- pure activation math). These are the ones we promote to
# INT16. Everything of a different type (Conv/MatMul/Gemm, i.e. the actual
# weighted layers) is left at INT8.
SCAN_LAYER_NAME_RE = re.compile(r"^/mamba_layers\.\d+/")
WEIGHTED_OP_TYPES = {"Conv", "MatMul", "Gemm", "ConvTranspose"}


# --------------------------------------------------------------------------
# Step 0: dataset
# --------------------------------------------------------------------------
class KWSPickleDataset(Dataset):
    """Loads the (X, y) MFCC pickles directly -- no dependency on the
    original training package."""

    def __init__(self, path):
        import pickle

        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.X = torch.from_numpy(data["X"]).float()  # (N, 49, 40)
        self.y = torch.from_numpy(data["y"]).long()  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# --------------------------------------------------------------------------
# Step 1: patch the ONNX model
# --------------------------------------------------------------------------
def patch_keepdims(onnx_path: Path, out_path: Path) -> Path:
    """esp-ppq's exporter reads node.attributes['keepdims'] directly for
    every Reduce* op and throws a KeyError if it's absent, instead of
    falling back to the ONNX-spec default of 1. Some exporters (including
    whatever produced this model's two RMSNorm layers) omit the attribute
    when it's left at its default value, which is valid ONNX but breaks
    the exporter. We add it back explicitly; this changes nothing about
    the model's behavior since 1 is what would have been assumed anyway.
    """
    m = onnx.load(str(onnx_path))
    reduce_types = {
        "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceL1",
        "ReduceL2", "ReduceProd", "ReduceLogSum", "ReduceLogSumExp",
        "ReduceSumSquare",
    }
    n_fixed = 0
    for n in m.graph.node:
        if n.op_type in reduce_types:
            if "keepdims" not in {a.name for a in n.attribute}:
                n.attribute.append(helper.make_attribute("keepdims", 1))
                n_fixed += 1
    if n_fixed:
        print(f"[patch] added explicit keepdims=1 to {n_fixed} Reduce* node(s)")
    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    return out_path


# --------------------------------------------------------------------------
# Step 2: figure out which ops to promote to INT16
# --------------------------------------------------------------------------
def get_scan_op_names(onnx_path: Path) -> list:
    """All ops belonging to the Mamba scan block(s), excluding the actual
    weighted layers (conv1d, in_proj, x_proj, dt_proj, out_proj), which
    stay INT8. Matches any number of mamba_layers.N (not just layer 0)."""
    m = onnx.load(str(onnx_path))
    names = [
        n.name
        for n in m.graph.node
        if SCAN_LAYER_NAME_RE.match(n.name) and n.op_type not in WEIGHTED_OP_TYPES
    ]
    return names


# --------------------------------------------------------------------------
# Step 3: quantize
# --------------------------------------------------------------------------
def quantize(onnx_path, espdl_path, calib_loader, calib_steps, device,
             int16_scan=True, tqt=False, tqt_block_size=4, tqt_steps=100,
             tqt_lr=1e-5, calib_algorithm="percentile", equalization=False,
             custom_scan_blocks=None):
    """
    int16_scan : if True (default, original behavior), promote the Mamba
        scan's elementwise ops to INT16 (mixed-precision config). If False,
        every op stays INT8 (plain all-int8 config) -- this is what we need
        to isolate whether TQT alone, with a wider block_size, can recover
        accuracy without paying the INT16 memory cost.
    tqt : if True, add a TrainedQuantizationThresholdPass stage (via
        setting.tqt_optimization) after calibration, with block_size /
        steps / lr as given below instead of TQTSetting's defaults
        (block_size=4, steps=500, lr=1e-5). Note esp-ppq's default block
        anchors are the graph's `is_computing_op` ops only (Conv/Gemm/
        MatMul/etc, i.e. the 7 weighted layers here) -- see block-log
        output for whether a given block_size actually reaches into the
        scan or not.
    calib_algorithm : 'percentile' | 'kl' | 'mse' | 'minmax'. Note
        espdl_setting()'s own *default* is actually 'kl', not 'percentile'
        -- we used to hardcode 'percentile' here because 'kl' + INT16
        mixed-precision hits a histogram-size bug (see REPORT.md 2c). That
        bug is specific to the INT8/INT16 boundary; for a plain all-INT8
        run there is no such boundary, so 'kl' should be usable here too.
    equalization : if True, enables esp-ppq's LayerwiseEqualizationPass
        (setting.equalization = True, top-level QuantizationSetting field).
        Note: this is NOT the same as tqt_optimization_setting.equalization
        -- the installed esp-ppq's TQTSetting class has no such field (its
        __init__ only defines interested_layers/lr/collecting_device/
        steps/is_scale_trainable/gamma/block_size/int_lambda), so setting
        that attribute directly (as the ESP-DL docs snippet suggests)
        would silently do nothing. The equalization flag that actually
        does something lives on the top-level setting object.
    custom_scan_blocks : if given (an int N), bypasses is_computing_op
        block-anchoring entirely and builds N blocks tiling the WHOLE
        graph end-to-end, split at N-1 evenly-spaced ReduceSum ops (one
        per unrolled SSM timestep) -- i.e. blocks whose boundaries sit
        *inside* the scan, not just around the 7 weighted ops. Requires
        `tqt=True`; tqt_block_size is ignored in this mode.
    """
    scan_names = get_scan_op_names(onnx_path)
    if int16_scan:
        print(f"[quantize] promoting {len(scan_names)} scan ops to INT16, "
              f"rest of the graph stays INT8")
    else:
        print(f"[quantize] all-INT8 config -- {len(scan_names)} scan ops "
              f"(that would have been promoted to INT16) are left at INT8")

    setting = QuantizationSettingFactory.espdl_setting()
    print(f"[quantize] calib_algorithm={calib_algorithm}")
    setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    setting.quantize_parameter_setting.calib_algorithm = calib_algorithm

    if equalization:
        print("[quantize] weight equalization (LayerwiseEqualizationPass) enabled")
        setting.equalization = True

    if int16_scan:
        for name in scan_names:
            setting.dispatching_table.append(name, TargetPlatform.ESPDL_S3_INT16)

    if tqt:
        print(f"[quantize] TQT enabled: block_size={tqt_block_size}, "
              f"steps={tqt_steps}, lr={tqt_lr}, collecting_device={device}")
        setting.tqt_optimization = True
        setting.tqt_optimization_setting.block_size = tqt_block_size
        setting.tqt_optimization_setting.steps = tqt_steps
        setting.tqt_optimization_setting.lr = tqt_lr
        setting.tqt_optimization_setting.collecting_device = device

    global _CUSTOM_BLOCK_BOUNDARY_NAMES
    if custom_scan_blocks:
        assert tqt, "custom_scan_blocks requires tqt=True"
        set_custom_scan_blocks(custom_scan_blocks, onnx_path)
    else:
        _CUSTOM_BLOCK_BOUNDARY_NAMES = None

    def collate_fn(batch):
        x, _ = batch
        return x.to(device)

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=INPUT_SHAPE,
        target="esp32s3",
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=setting,
        device=device,
        error_report=False,
        verbose=0,
        skip_export=False,
    )
    return quant_graph


# --------------------------------------------------------------------------
# Step 4: evaluate
# --------------------------------------------------------------------------
def eval_float(onnx_path, ds, device, max_samples=None, seed=123):
    rng = np.random.default_rng(seed)
    if max_samples and max_samples < len(ds):
        idx = rng.choice(len(ds), size=max_samples, replace=False)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    sess = ort.InferenceSession(str(onnx_path))
    preds, labels = [], []
    for x, y in loader:
        out = sess.run(None, {"input": x.numpy()})[0]
        preds.append(np.argmax(out, axis=1))
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return float((preds == labels).mean() * 100.0), len(ds)


def eval_quant(graph, ds, device, max_samples=None, seed=123):
    rng = np.random.default_rng(seed)
    if max_samples and max_samples < len(ds):
        idx = rng.choice(len(ds), size=max_samples, replace=False)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    executor = TorchExecutor(graph=graph, device=device)
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = executor.forward(inputs=x)[0].detach().cpu().numpy()
        preds.append(np.argmax(out, axis=1))
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return float((preds == labels).mean() * 100.0), len(ds)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True, help="path to the original ONNX model")
    ap.add_argument("--val", type=Path, required=True, help="val.pkl, used ONLY for calibration")
    ap.add_argument("--test", type=Path, required=True, help="test.pkl, used ONLY for accuracy evaluation")
    ap.add_argument("--output", type=Path, default=None, help="output .espdl path")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-steps", type=int, default=256)
    ap.add_argument("--eval-samples", type=int, default=0, help="0 = full test set")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--int16-scan", action=argparse.BooleanOptionalAction, default=True,
                     help="promote scan ops to INT16 (mixed-precision, original default). "
                          "Use --no-int16-scan for the plain all-int8 config.")
    ap.add_argument("--tqt", action="store_true", default=False,
                     help="enable TrainedQuantizationThresholdPass after calibration")
    ap.add_argument("--tqt-block-size", type=int, default=4,
                     help="TQTSetting.block_size (esp-ppq default is 4)")
    ap.add_argument("--tqt-steps", type=int, default=100)
    ap.add_argument("--tqt-lr", type=float, default=1e-5)
    ap.add_argument("--calib-algorithm", type=str, default="percentile",
                     choices=["percentile", "kl", "mse", "minmax"])
    ap.add_argument("--equalization", action="store_true", default=False,
                     help="enable esp-ppq LayerwiseEqualizationPass (setting.equalization)")
    ap.add_argument("--custom-scan-blocks", type=int, default=None,
                     help="if set to N, bypass is_computing_op block anchoring and build "
                          "N blocks tiling the whole graph, split at N-1 evenly-spaced "
                          "ReduceSum ops inside the SSM scan. Requires --tqt.")
    args = ap.parse_args()

    if args.output is None:
        args.output = args.onnx.with_name(f"{args.onnx.stem}-int8.espdl")

    work_dir = args.output.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: patch ----
    fixed_onnx = work_dir / f"{args.onnx.stem}-fixed.onnx"
    patch_keepdims(args.onnx, fixed_onnx)

    # ---- Step 2: data ----
    val_ds = KWSPickleDataset(args.val)
    test_ds = KWSPickleDataset(args.test)
    print(f"val (calibration pool) = {len(val_ds)} samples")
    print(f"test (evaluation, held out) = {len(test_ds)} samples")

    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(len(val_ds), size=min(args.calib_samples, len(val_ds)), replace=False)
    calib_loader = DataLoader(Subset(val_ds, calib_idx.tolist()), batch_size=1, shuffle=False)

    # ---- Step 3: float baseline ----
    print("\n[1/3] Float accuracy (ONNX Runtime) ...")
    t0 = time.time()
    float_acc, n = eval_float(fixed_onnx, test_ds, args.device, max_samples=args.eval_samples or None)
    print(f"  {float_acc:.2f}%  (n={n}, {time.time()-t0:.1f}s)")

    # ---- Step 4: quantize ----
    mode_str = "INT8 + INT16 scan" if args.int16_scan else "all-INT8"
    tqt_str = (f", TQT block_size={args.tqt_block_size} steps={args.tqt_steps} lr={args.tqt_lr}"
               if args.tqt else ", no TQT")
    print(f"\n[2/3] Quantizing ({mode_str}{tqt_str}, calib={args.calib_algorithm}, "
          f"equalization={args.equalization}, custom_scan_blocks={args.custom_scan_blocks}, "
          f"target=esp32s3) ...")
    t0 = time.time()
    quant_graph = quantize(
        fixed_onnx, args.output, calib_loader, args.calib_steps, args.device,
        int16_scan=args.int16_scan, tqt=args.tqt, tqt_block_size=args.tqt_block_size,
        tqt_steps=args.tqt_steps, tqt_lr=args.tqt_lr,
        calib_algorithm=args.calib_algorithm, equalization=args.equalization,
        custom_scan_blocks=args.custom_scan_blocks,
    )
    print(f"  done in {time.time()-t0:.1f}s -> {args.output}")

    # ---- Step 5: quantized accuracy ----
    print(f"\n[3/3] Quantized accuracy (esp-ppq TorchExecutor) ...")
    t0 = time.time()
    quant_acc, n = eval_quant(quant_graph, test_ds, args.device, max_samples=args.eval_samples or None)
    print(f"  {quant_acc:.2f}%  (n={n}, {time.time()-t0:.1f}s)")

    size_kb = args.output.stat().st_size / 1024 if args.output.exists() else 0
    print(f"\n{'='*60}")
    print(f"  Float accuracy:      {float_acc:.2f}%")
    print(f"  Quantized accuracy:  {quant_acc:.2f}%  (drop: {float_acc - quant_acc:.2f}pp)")
    print(f"  Output file:         {args.output}  ({size_kb:.1f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
