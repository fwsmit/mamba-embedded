#!/usr/bin/env python3
"""
Quick standalone test of espdl_quantize_onnx for a KWS ONNX model.

Usage:
    python test_quantize.py path/to/model.onnx

This avoids the full pipeline in quantize.py (no validation evaluation,
no dataset export, no model size report). Just calibrates with 10 samples
and writes the .espdl file next to the ONNX file.

NOTE: Uses espdl_quantize_onnx (single quantization), NOT espdl_auto_quantize_onnx
      (AutoQuant search). The auto variant defaults to an exhaustive search over
      1,067,040 parameter combinations and would take days to complete.
      See the docstring in esp_ppq/autoquant/interface.py for details.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from esp_ppq.api import espdl_quantize_onnx

# Add project root to path so we can import from train/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train.data import load_speechcommands_data


def main():
    parser = argparse.ArgumentParser(
        description="Quick quantization test for KWS ONNX models"
    )
    parser.add_argument("onnx_path", type=Path, help="Path to the ONNX model file")
    parser.add_argument(
        "--calib-samples", type=int, default=10,
        help="Number of calibration samples (default: 10)"
    )
    parser.add_argument(
        "--calib-steps", type=int, default=10,
        help="Calibration steps (default: 10)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for PPQ calibration (default: auto)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .espdl path (default: same stem as ONNX)"
    )
    args = parser.parse_args()

    onnx_path = args.onnx_path.resolve()
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found", file=sys.stderr)
        sys.exit(1)

    # Infer input shape from ONNX
    import onnx
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    input_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"ONNX input shape: {input_shape}")

    # Determine dataset from shape
    shape = tuple(input_shape)
    if shape == (1, 49, 40):
        dataset_name = "kws"
    elif shape == (1, 10, 57):
        dataset_name = "har"
    else:
        print(f"WARNING: unknown input shape {shape}, assuming KWS")
        dataset_name = "kws"

    # Load calibration data
    data_dir = Path.home() / "Datasets"
    if dataset_name == "kws":
        print(f"Loading KWS calibration data from {data_dir} ...")
        train_ds = load_speechcommands_data(str(data_dir), split="train")
    else:
        from train.data import load_har_data
        print(f"Loading HAR calibration data from {data_dir} ...")
        train_ds = load_har_data(data_dir, split="train")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_ds), size=min(args.calib_samples, len(train_ds)), replace=False)
    calib_ds = Subset(train_ds, indices.tolist())

    device = args.device

    def collate_fn(batch):
        x, _ = batch
        return x.to(device)

    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, drop_last=False)

    # Output path
    if args.output:
        espdl_path = args.output.resolve()
    else:
        espdl_path = onnx_path.with_suffix(".espdl")

    espdl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running quantization on {device} ...")
    print(f"  ONNX:      {onnx_path}")
    print(f"  Output:    {espdl_path}")
    print(f"  Shape:     {input_shape}")
    print(f"  Calib:     {len(calib_ds)} samples, {args.calib_steps} steps")
    print()

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target="esp32s3",
        num_of_bits=8,
        collate_fn=collate_fn,
        device=device,
        error_report=True,
        verbose=0,
    )

    print()
    print(f"✓ Done — {espdl_path}")
    print(f"  Size: {espdl_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()