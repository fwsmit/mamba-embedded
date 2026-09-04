"""Build a deployable .espdl + dataset.bin for a given int16 dispatch list,
named so run-esp.sh can pick them up (trial-N pattern), then report latency.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path("/home/friso/Agents/mamba-embedded")
PY = "/home/friso/.conda/envs/torch-pascal/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--calib-alg", default="percentile")
    ap.add_argument("--calib-steps", type=int, default=256)
    ap.add_argument("--int16-file", required=True)
    ap.add_argument("--tqt", action="store_true")
    ap.add_argument("--outdir", default=str(REPO / "out" / "deploy"))
    ap.add_argument("--datasets", default=str(REPO / "kws_dataset"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # model + dataset names so run-esp.sh regex matches: trial-<N><suffix>.espdl
    espdl = outdir / f"model-trial-{args.trial}{args.suffix}.espdl"
    dbin = outdir / f"dataset-trial-{args.trial}{args.suffix}.bin"
    int16 = json.load(open(args.int16_file))

    sys.path.insert(0, str(REPO))
    from train.kws_qsearch import (
        load_calib_loader, load_val, quantize, quantize_dataset_to_bin,
        get_input_quantization, collate_loader_fn,
    )
    device = "cpu"
    calib = load_calib_loader(args.calib_steps)
    val_ds, _ = load_val()

    qg = quantize(
        args.onnx, espdl, calib, args.calib_steps, args.calib_alg,
        args.tqt, 512, 3000, 2e-4, 8, int16, device,
    )
    configs = get_input_quantization(qg)
    quantize_dataset_to_bin(configs, val_ds, dbin)
    print(f"built {espdl} and {dbin}")
    print(espdl)


if __name__ == "__main__":
    main()