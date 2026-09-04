"""Build deployable .espdl + dataset.bin bundles quantized entirely at 16-bit,
named so run-esp.sh can pick them up (trial-N pattern), for latency comparison
against the mixed 8/16-bit deploy.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/friso/Agents/mamba-embedded")
PY = "/home/friso/.conda/envs/torch-pascal/bin/python"


def patch_keepdims(onnx_path: Path) -> Path:
    import onnx
    m = onnx.load(str(onnx_path))
    changed = 0
    for n in m.graph.node:
        if n.op_type in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
                         "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
                         "ReduceProd", "ReduceSumSquare"):
            if not any(a.name == "keepdims" for a in n.attribute):
                n.attribute.add().CopyFrom(
                    onnx.helper.make_attribute("keepdims", 1))
                changed += 1
    if changed:
        out = onnx_path.with_name(onnx_path.stem + "_full16fix.onnx")
        onnx.save(m, str(out))
        return out
    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--suffix", default="_full16")
    ap.add_argument("--calib-alg", default="percentile")
    ap.add_argument("--calib-steps", type=int, default=256)
    ap.add_argument("--outdir", default=str(REPO / "out" / "deploy"))
    ap.add_argument("--datasets", default=str(REPO / "kws_dataset"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nsamples", type=int, default=None,
                    help="limit dataset to N samples (int16 dataset exceeds the \n9MB partition; latency is measured on the first sample)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    espdl = outdir / f"model-trial-{args.trial}{args.suffix}.espdl"
    dbin = outdir / f"dataset-trial-{args.trial}{args.suffix}.bin"
    onnx_path = patch_keepdims(Path(args.onnx))

    sys.path.insert(0, str(REPO))
    from train.kws_qsearch import (
        load_calib_loader, load_val, quantize, get_input_quantization,
    )
    from train.quantize_lite import export_int16_dataset_to_bin
    device = args.device
    calib = load_calib_loader(args.calib_steps)
    val_ds, _ = load_val()

    qg = quantize(
        onnx_path, espdl.with_suffix(".espdl"), calib, args.calib_steps,
        args.calib_alg, False, 512, 3000, 2e-4, 16, [], device,
    )
    configs = get_input_quantization(qg)
    if args.nsamples is not None:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, range(args.nsamples))
    export_int16_dataset_to_bin(configs, val_ds, dbin)
    print(f"built {espdl.with_suffix('.espdl')} and {dbin}")
    print(espdl.with_suffix(".espdl"))


if __name__ == "__main__":
    main()