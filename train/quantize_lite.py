#!/usr/bin/env python3
"""
Quantize the 7 fixed-architecture Mamba-Lite micro comparison models
(mambalite-micro/*.onnx) with every supported ESP-DL quantization method:

    standard    8-bit PTQ (32 uniform-random calibration samples)
    strat-kl-tqt  8-bit PTQ with KL calibration over 256 stratified samples
                 followed by a graph-wide TQT pass (best 8-bit config)
    int16      16-bit PTQ

and additionally export the unquantized FP32 model (if possible), so the
models can also be run on the ESP32-S3 as a float baseline.

For every variation a quantized dataset binary is written next to the .espdl
file (named dataset-<model><suffix>.bin) so run-esp.sh can flash it to the
ESP32-S3. All artefacts land in experiments/mambalite-micro/ and every
quantization attempt (including failures) is recorded in
quantization_manifest.json.

Usage:
    conda run -n torch-pascal python -m train.quantize_lite
    conda run -n torch-pascal python -m train.quantize_lite --models har-single kws-single
"""

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .quantize import (
    BEST_CALIB_STEPS,
    CALIB_BATCH,
    CALIB_STEPS,
    TARGET,
    QuantizationDivergedError,
    collate_fn,
    get_espdl_param_size,
    get_input_quantization,
    infer_input_shape,
    load_calibration,
    load_calibration_stratified,
    quantize_dataset_to_bin,
    quantize_onnx_to_espdl,
    quantize_onnx_to_espdl_best,
)
from .train_lite import _load_lite_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_SRC = REPO_ROOT / "mambalite-micro"
EXPERIMENTS = REPO_ROOT / "experiments" / "mambalite-micro"

# model name -> dataset (used for calibration / dataset binary generation)
MODEL_DATASETS = {
    "har-bidir-add": "har",
    "har-bidir-mul": "har",
    "har-single": "har",
    "kws-bidir-add": "kws",
    "kws-bidir-mul": "kws",
    "kws-single": "kws",
    "kws-bidir-mul-2": "kws",
}

# (method name, quantize fn, calibration kind, calib steps, bits, filename suffix)
METHODS = [
    ("standard", quantize_onnx_to_espdl, "uniform", CALIB_STEPS, 8, ""),
    ("strat-kl-tqt", quantize_onnx_to_espdl_best, "stratified", BEST_CALIB_STEPS, 8, "_strat"),
    ("int16", quantize_onnx_to_espdl, "uniform", CALIB_STEPS, 16, "_int16"),
]

DEVICE = "cpu"  # no CUDA available in this environment


def _patch_esp_ppq_input_limit():
    """Widen esp_ppq's defensive per-op input cap.

    The TorchExecutor's ``ASSERT_NUM_OF_INPUT`` defaults to at most 99 inputs
    per op. The unrolled Mamba selective-scan of the seq-len-100 KWS models
    emits a Concat with 100 inputs, which trips that cap. The env's esp_ppq is
    installed on a read-only filesystem, so the same one-line change
    (99 -> 1024) is applied at runtime.
    """
    import esp_ppq.executor.op.torch.base as _base
    import esp_ppq.executor.op.torch.cuda as _cuda
    import esp_ppq.executor.op.torch.default as _default

    def ASSERT_NUM_OF_INPUT(op, values, min_num_of_input=-1, max_num_of_input=1024):  # noqa: N802
        if min_num_of_input == max_num_of_input:
            if len(values) != min_num_of_input:
                raise ValueError(
                    f"Can not feed value to operation {op.name}, "
                    f"expects exact {min_num_of_input} inputs, however {len(values)} was given"
                )
        elif len(values) > max_num_of_input:
            raise ValueError(
                f"Too many input value for {op.name}, "
                f"expects {max_num_of_input} inputs at most, however {len(values)} was given"
            )
        elif len(values) < min_num_of_input:
            raise ValueError(
                f"Too few input value for {op.name}, "
                f"expects {min_num_of_input} inputs at least, however {len(values)} was given"
            )

    _base.ASSERT_NUM_OF_INPUT = ASSERT_NUM_OF_INPUT
    _default.ASSERT_NUM_OF_INPUT = ASSERT_NUM_OF_INPUT
    _cuda.ASSERT_NUM_OF_INPUT = ASSERT_NUM_OF_INPUT


_patch_esp_ppq_input_limit()


def build_calib_loader(dataset, train_lite, kind: str):
    """Calibration loader for a method, derived from the shared lite train set."""
    if kind == "stratified":
        return load_calibration_stratified(
            dataset, REPO_ROOT, n_samples=BEST_CALIB_STEPS, train_ds=train_lite
        )
    return load_calibration(
        dataset, REPO_ROOT, CALIB_STEPS * CALIB_BATCH, train_ds=train_lite
    )


def export_int16_dataset_to_bin(configs, dataset, output_path):
    """Quantize a dataset to int16 using the model's input TQC.

    Mirrors ``quantize_dataset_to_bin`` (same header layout, same PPQ formula
    q = clip(round(x/scale - offset), quant_min, quant_max)) but stores int16
    samples, matching the INT16 input tensor of 16-bit .espdl models.
    """
    tqc = next((c for c in configs.values() if c is not None), None)
    if tqc is None:
        print("WARNING: No quantized input config found — skipping dataset quantization.")
        return

    scale = tqc.scale.cpu()
    offset = tqc.offset.cpu()
    quant_min = tqc.quant_min
    quant_max = tqc.quant_max

    all_quant = []
    elements_per_sample = None
    for i, (features, _label) in enumerate(dataset):
        if features.dim() == 2:
            x = features.unsqueeze(0)  # (1, seq_len, feat_dim)
        else:
            x = features
        q = torch.clamp(torch.round(x / scale - offset), quant_min, quant_max).to(torch.int16)
        flat = q.flatten()
        if elements_per_sample is None:
            elements_per_sample = flat.numel()
        all_quant.append(flat)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(all_quant)))
        f.write(struct.pack("<I", elements_per_sample))
        for q in all_quant:
            f.write(q.numpy().tobytes())
    size_kb = output_path.stat().st_size / 1024
    print(f"  Wrote {len(all_quant)} samples x {elements_per_sample} int16 values -> {output_path} ({size_kb:.1f} KB)")


def write_dataset_bin(configs, dataset, output_path, num_of_bits):
    """Write the dataset binary matching the model's input tensor width."""
    if num_of_bits == 16:
        export_int16_dataset_to_bin(configs, dataset, output_path)
    else:
        quantize_dataset_to_bin(configs, dataset, output_path)


def export_float_dataset_to_bin(dataset, output_path):
    """Write a dataset as raw float32 (for the unquantized FP32 models).

    Layout matches quantize_dataset_to_bin: uint32 num_samples, uint32
    elements_per_sample, then float32 samples flattened row-major.
    """
    all_flat = []
    elements_per_sample = None
    for features, _label in dataset:
        if features.dim() == 2:
            x = features.unsqueeze(0)
        else:
            x = features
        flat = x.float().flatten()
        if elements_per_sample is None:
            elements_per_sample = flat.numel()
        all_flat.append(flat)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(all_flat)))
        f.write(struct.pack("<I", elements_per_sample))
        for q in all_flat:
            f.write(q.numpy().astype(np.float32).tobytes())
    size_kb = output_path.stat().st_size / 1024
    print(f"  Wrote {len(all_flat)} samples x {elements_per_sample} float32 -> {output_path} ({size_kb:.1f} KB)")


def quantize_float(onnx_copy, espdl_path, input_shape):
    """Export an unquantized FP32 .espdl model (via esp_ppq's FP32 branch)."""
    from esp_ppq.api import espdl_quantize_onnx

    ds = torch.utils.data.TensorDataset(
        torch.zeros(1, *input_shape[1:]), torch.zeros(1, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    espdl_quantize_onnx(
        onnx_import_file=str(onnx_copy),
        espdl_export_file=str(espdl_path),
        calib_dataloader=loader,
        calib_steps=1,
        input_shape=input_shape,
        target=TARGET,
        num_of_bits=8,
        float=True,
        device=DEVICE,
        collate_fn=collate_fn,
        export_test_values=False,
        error_report=False,
        skip_export=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only quantize these model names (default: all 7)")
    parser.add_argument("--skip-float", action="store_true",
                        help="Skip the unquantized FP32 export")
    parser.add_argument("--out", type=Path, default=EXPERIMENTS,
                        help="Output directory (default: experiments/mambalite-micro)")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "quantization_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    train_cache: dict = {}
    for name, dataset in MODEL_DATASETS.items():
        if args.models and name not in args.models:
            continue
        if dataset not in train_cache:
            print(f"Loading Mamba-Lite {dataset} data ...", flush=True)
            train_cache[dataset] = _load_lite_data(dataset, str(Path.home() / "Datasets"))
        train_lite, val_lite = train_cache[dataset]

        src_onnx = MODELS_SRC / f"{name}.onnx"
        if not src_onnx.exists():
            print(f"ERROR: source ONNX not found: {src_onnx}", file=sys.stderr)
            sys.exit(1)

        # Copy first: esp_ppq simplifies the file it is given in-place, and the
        # models in mambalite-micro/ are the pristine training outputs.
        onnx_copy = out_dir / f"{name}.onnx"
        shutil.copy2(src_onnx, onnx_copy)
        input_shape = infer_input_shape(onnx_copy)

        model_entry = manifest.setdefault(name, {})
        model_entry["dataset"] = dataset
        model_entry["input_shape"] = input_shape

        print(f"\n=== {name} (dataset={dataset}, input={input_shape}) ===", flush=True)

        for method_name, quantize_fn, calib_kind, calib_steps, bits, suffix in METHODS:
            espdl_path = out_dir / f"{name}{suffix}.espdl"
            if model_entry.get(method_name, {}).get("status") == "ok" and espdl_path.exists():
                print(f"  [{method_name}] already quantized, skipping", flush=True)
                continue

            print(f"  [{method_name}] quantizing ...", flush=True)
            t0 = time.time()
            try:
                calib = build_calib_loader(dataset, train_lite, calib_kind)
                quant_graph = quantize_fn(
                    onnx_path=onnx_copy,
                    espdl_path=espdl_path,
                    calib_loader=calib,
                    calib_steps=calib_steps,
                    input_shape=input_shape,
                    target=TARGET,
                    num_of_bits=bits,
                    device=DEVICE,
                    collate_fn=collate_fn,
                )
                configs = get_input_quantization(quant_graph)
                write_dataset_bin(
                    configs, val_lite, out_dir / f"dataset-{name}{suffix}.bin", bits
                )
                model_entry[method_name] = {
                    "status": "ok",
                    "seconds": round(time.time() - t0, 1),
                    "espdl_bytes": espdl_path.stat().st_size,
                    "param_bytes": get_espdl_param_size(espdl_path.with_suffix(".info")),
                }
                print(f"  [{method_name}] OK ({model_entry[method_name]['seconds']}s, "
                      f"{espdl_path.stat().st_size} bytes)", flush=True)
            except QuantizationDivergedError as e:
                model_entry[method_name] = {
                    "status": "failed", "error": f"QuantizationDivergedError: {e}",
                    "seconds": round(time.time() - t0, 1),
                }
                print(f"  [{method_name}] FAILED (diverged): {e}", flush=True)
            except Exception as e:  # noqa: BLE001 — record any quantization failure
                model_entry[method_name] = {
                    "status": "failed", "error": f"{type(e).__name__}: {e}",
                    "seconds": round(time.time() - t0, 1),
                }
                print(f"  [{method_name}] FAILED: {type(e).__name__}: {e}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

        if not args.skip_float:
            espdl_path = out_dir / f"{name}_float.espdl"
            if model_entry.get("float", {}).get("status") == "ok" and espdl_path.exists():
                print("  [float] already exported, skipping", flush=True)
            else:
                print("  [float] exporting unquantized FP32 ...", flush=True)
                t0 = time.time()
                try:
                    quantize_float(onnx_copy, espdl_path, input_shape)
                    export_float_dataset_to_bin(val_lite, out_dir / f"dataset-{name}_float.bin")
                    model_entry["float"] = {
                        "status": "ok",
                        "seconds": round(time.time() - t0, 1),
                        "espdl_bytes": espdl_path.stat().st_size,
                        "param_bytes": get_espdl_param_size(espdl_path.with_suffix(".info")),
                    }
                    print(f"  [float] OK ({model_entry['float']['seconds']}s, "
                          f"{espdl_path.stat().st_size} bytes)", flush=True)
                except Exception as e:  # noqa: BLE001
                    model_entry["float"] = {
                        "status": "failed", "error": f"{type(e).__name__}: {e}",
                        "seconds": round(time.time() - t0, 1),
                    }
                    print(f"  [float] FAILED: {type(e).__name__}: {e}", flush=True)
                manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nQuantization finished. Manifest: {manifest_path}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()