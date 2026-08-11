#!/usr/bin/env python3
"""
Quick standalone test of espdl_quantize_onnx for KWS/HAR ONNX models.

Usage:
    python test_quantize.py path/to/model.onnx [options]

Quantizes the model and evaluates both float (ONNX Runtime) and quantized
(TorchExecutor) accuracy on a subset of the validation set.

Options include calibration algorithm selection, equalization, bias correction,
LSQ, TQT (Trained Quantization Thresholds), and bit-width selection (8 or 16).

Best results for Mamba models use 16-bit quantization with percentile calibration,
which achieves ~87% accuracy (vs 92% float) with negligible model size increase.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.executor import TorchExecutor

# Add project root to path so we can import from train/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train.data import load_speechcommands_data


def evaluate_accuracy(
    model_or_graph,
    val_ds,
    device: str,
    max_samples: int = 0,
    is_quantized: bool = False,
):
    """
    Evaluate accuracy of a model on a dataset (subsampled).

    For float models, model_or_graph is an ONNX path (str).
    For quantized models, model_or_graph is a BaseGraph for TorchExecutor.
    """
    rng = np.random.default_rng(42)
    if max_samples > 0 and max_samples < len(val_ds):
        indices = rng.choice(len(val_ds), size=max_samples, replace=False)
        eval_ds = Subset(val_ds, indices.tolist())
    else:
        eval_ds = val_ds

    loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    all_preds = []
    all_labels = []

    if is_quantized:
        executor = TorchExecutor(graph=model_or_graph, device=device)
        for x, y in loader:
            x = x.to(device)
            out = executor.forward(inputs=x)[0].cpu().numpy()
            preds = np.argmax(out, axis=1)
            all_preds.append(preds)
            all_labels.append(y.numpy())
    else:
        ort_sess = ort.InferenceSession(str(model_or_graph))
        for x, y in loader:
            out = ort_sess.run(None, {"input": x.numpy()})[0]
            preds = np.argmax(out, axis=1)
            all_preds.append(preds)
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.mean(all_preds == all_labels) * 100.0
    return acc, len(eval_ds)


def main():
    parser = argparse.ArgumentParser(
        description="Quick quantization test for KWS/HAR ONNX models"
    )
    parser.add_argument("onnx_path", type=Path, help="Path to the ONNX model file")
    parser.add_argument(
        "--calib-samples", type=int, default=128,
        help="Number of calibration samples (default: 128)"
    )
    parser.add_argument(
        "--calib-steps", type=int, default=128,
        help="Calibration steps (default: 128)"
    )
    parser.add_argument(
        "--bits", type=int, default=16, choices=[8, 16],
        help="Quantization bit-width (default: 16, best for Mamba models)"
    )
    parser.add_argument(
        "--val-samples", type=int, default=500,
        help="Number of validation samples for accuracy eval (default: 500, 0=all)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for PPQ calibration (default: auto)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .espdl path (default: <stem>-<bits>bit.espdl)"
    )
    parser.add_argument(
        "--calib-algo", type=str, default="percentile",
        choices=["minmax", "kl", "percentile", "mse"],
        help="Calibration algorithm for activations (default: percentile)"
    )
    parser.add_argument(
        "--equalization", action="store_true", default=False,
        help="Enable layer-wise equalization"
    )
    parser.add_argument(
        "--equalization-iter", type=int, default=5,
        help="Equalization iterations (default: 5)"
    )
    parser.add_argument(
        "--bias-correct", action="store_true", default=False,
        help="Enable bias correction"
    )
    parser.add_argument(
        "--tqt", action="store_true", default=False,
        help="Enable TQT optimization (TrainedQuantizationThresholds)"
    )
    parser.add_argument(
        "--tqt-steps", type=int, default=200,
        help="TQT training steps (default: 200)"
    )
    parser.add_argument(
        "--tqt-lr", type=float, default=1e-5,
        help="TQT learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose output"
    )
    args = parser.parse_args()

    onnx_path = args.onnx_path.resolve()
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found", file=sys.stderr)
        sys.exit(1)

    device = args.device

    # Infer input shape from ONNX
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

    # Load data
    data_dir = Path.home() / "Datasets"
    if dataset_name == "kws":
        print(f"Loading KWS data from {data_dir} ...")
        train_ds = load_speechcommands_data(str(data_dir), split="train")
        val_ds = load_speechcommands_data(str(data_dir), split="val")
    else:
        from train.data import load_har_data
        print(f"Loading HAR data from {data_dir} ...")
        train_ds = load_har_data(data_dir, split="train")
        val_ds = load_har_data(data_dir, split="val")

    print(f"  Training samples:   {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")

    # -----------------------------------------------------------------------
    # Step 0: Float accuracy
    # -----------------------------------------------------------------------
    print("\n[0/3] Measuring float accuracy (ONNX Runtime) ...")
    t0 = time.time()
    float_acc, n_val = evaluate_accuracy(
        str(onnx_path), val_ds, device,
        max_samples=args.val_samples,
    )
    t_float = time.time() - t0
    print(f"  Float accuracy: {float_acc:.2f}%  (n={n_val}, {t_float:.1f}s)")

    # -----------------------------------------------------------------------
    # Step 1: Calibration data
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_calib = min(args.calib_samples, len(train_ds))
    indices = rng.choice(len(train_ds), size=n_calib, replace=False)
    calib_ds = Subset(train_ds, indices.tolist())

    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, drop_last=False)

    def collate_fn(batch):
        x, _ = batch
        return x.to(device)

    # Output path
    if args.output:
        espdl_path = args.output.resolve()
    else:
        espdl_path = onnx_path.with_name(f"{onnx_path.stem}-{args.bits}bit.espdl")

    espdl_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 2: Quantization
    # -----------------------------------------------------------------------
    print(f"\n[1/3] Running PTQ on {device} ({args.bits}-bit) ...")
    print(f"  ONNX:      {onnx_path}")
    print(f"  Output:    {espdl_path}")
    print(f"  Shape:     {input_shape}")
    print(f"  Calib:     {n_calib} samples, {args.calib_steps} steps")

    # Build custom setting based on ESPDL default
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = args.calib_algo
    print(f"  Calib algo: {args.calib_algo}")

    if args.equalization:
        print(f"  Equalization: ON (iter={args.equalization_iter})")
        setting.equalization = True
        setting.equalization_setting.iterative = args.equalization_iter
        setting.equalization_setting.value_threshold = 0.5

    if args.bias_correct:
        print(f"  Bias correction: ON")
        setting.bias_correct = True
        setting.bias_correct_setting.num_steps = 4

    if args.tqt:
        print(f"  TQT optimization: ON (steps={args.tqt_steps}, lr={args.tqt_lr})")
        setting.tqt_optimization = True
        setting.tqt_optimization_setting.steps = args.tqt_steps
        setting.tqt_optimization_setting.lr = args.tqt_lr
        setting.tqt_optimization_setting.collecting_device = device

    t0 = time.time()
    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target="esp32s3",
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=setting,
        device=device,
        error_report=args.verbose,
        verbose=1 if args.verbose else 0,
        skip_export=False,
    )
    t_quant = time.time() - t0
    print(f"  Quantization done in {t_quant:.1f}s")

    # -----------------------------------------------------------------------
    # Step 3: Quantized accuracy
    # -----------------------------------------------------------------------
    print(f"\n[2/3] Measuring quantized accuracy (TorchExecutor) ...")
    t0 = time.time()
    quant_acc, n_val_q = evaluate_accuracy(
        quant_graph, val_ds, device,
        max_samples=args.val_samples,
        is_quantized=True,
    )
    t_quant_eval = time.time() - t0
    print(f"  Quantized accuracy: {quant_acc:.2f}%  (n={n_val_q}, {t_quant_eval:.1f}s)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    drop = float_acc - quant_acc
    size_kb = espdl_path.stat().st_size / 1024 if espdl_path.exists() else 0

    print(f"\n{'='*65}")
    print(f"  Summary for {onnx_path.name}")
    print(f"  {'-'*61}")
    print(f"  {'Bits:':27s} {args.bits}-bit")
    print(f"  {'Calib algo:':27s} {args.calib_algo}")
    print(f"  {'Enhancements:':27s} "
          f"eqz={'Y' if args.equalization else 'N'}, "
          f"bc={'Y' if args.bias_correct else 'N'}, "
          f"tqt={'Y' if args.tqt else 'N'}")
    print(f"  {'Float accuracy:':27s} {float_acc:.2f}%")
    print(f"  {'Quantized accuracy:':27s} {quant_acc:.2f}%")
    print(f"  {'Accuracy drop:':27s} {drop:.2f}pp")
    print(f"  {'Model size:':27s} {size_kb:.1f} KB")
    print(f"  {'Total time:':27s} {t_quant + t_quant_eval:.1f}s")
    print(f"{'='*65}")
    print(f"\n  Output: {espdl_path}")

    return float_acc, quant_acc


if __name__ == "__main__":
    main()