from esp_ppq.api import export_ppq_graph
from esp_ppq.IR import BaseGraph, Operation
# from esp_ppq.passes import QuantizationOptimizationPass
from esp_ppq.core import TargetPlatform
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import Subset
from .data import load_har_data, load_speechcommands_data
from code import InteractiveConsole

from esp_ppq.executor import TorchExecutor
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "esp32s3"
NUM_OF_BITS = 8
CALIB_STEPS = 32
CALIB_BATCH = 1

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a Mamba ONNX model to ESP-DL .espdl format"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["har", "kws"],
        help="Dataset the model was trained on",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["mamba-1", "mamba-1-sim", "mamba-3"],
        help="Model architecture",
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=NUM_OF_BITS,
        choices=[8, 16],
        help="Quantisation bit-width (default: 8)",
    )

    parser.add_argument(
        "--calib-steps",
        type=int,
        default=CALIB_STEPS,
        help="Number of calibration batches (default: 32)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for PPQ calibration: 'cpu' or 'cuda' (default: cpu)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_datasets(dataset: str):
    """
    Load all dataset splits for a given dataset.

    Returns (train_ds, val_ds, test_ds).
    """
    data_dir = Path.home() / "Datasets"

    if dataset == "har":
        return load_har_data(data_dir)

    elif dataset == "kws":
        return load_speechcommands_data(data_dir=str(data_dir))

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def infer_input_shape(dataset: str):
    if dataset == "har":
        # HAR ONNX input:
        # [batch, time, features]
        return [1, 10, 57]

    elif dataset == "kws":
        # KWS ONNX input:
        # [batch, time, mfcc]
        return [1, 49, 40]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Calibration dataloader
# ---------------------------------------------------------------------------


DEVICE = "cpu"


def collate_fn(batch):
    x, _ = batch
    return x.to("cpu")


def load_calibration(
    dataset: str,
    root: Path,
    n_samples: int,
):
    """
    Build calibration dataloader from existing datasets.
    """

    train_ds, _, _ = load_datasets(dataset)

    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.choice(
        len(train_ds),
        size=min(n_samples, len(train_ds)),
        replace=False,
    )

    calib_ds = Subset(train_ds, indices.tolist())

    loader = DataLoader(
        calib_ds,
        batch_size=CALIB_BATCH,
        shuffle=False,
        drop_last=False,
    )

    return loader


# ---------------------------------------------------------------------------
# Quantization loss evaluation
# ---------------------------------------------------------------------------


def _softmax(x):
    """Numerically stable softmax over the last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def evaluate_quantization_loss(
    quant_graph: BaseGraph,
    onnx_path: str,
    val_ds,
    device: str = "cpu",
):
    """
    Run float and quantized inference on the full validation set and report
    probability-level errors, accuracy drop, and per-class probability MSE.
    """
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    ort_sess = ort.InferenceSession(str(onnx_path))
    quant_executor = TorchExecutor(graph=quant_graph, device=device)

    all_float_logits = []
    all_quant_logits = []
    all_labels = []

    for data, target in val_loader:
        data_np = data.numpy()

        # Float inference
        float_out = ort_sess.run(None, {"input": data_np})[0]

        # Quantized inference
        quant_out = quant_executor.forward(inputs=data.to(device))[0]
        quant_out = quant_out.cpu().numpy()

        all_float_logits.append(float_out)
        all_quant_logits.append(quant_out)
        all_labels.append(target.numpy())

    all_float_logits = np.concatenate(all_float_logits, axis=0)
    all_quant_logits = np.concatenate(all_quant_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert to probabilities
    all_float_probs = _softmax(all_float_logits)
    all_quant_probs = _softmax(all_quant_logits)

    # --- Probability-level error ---
    prob_diff = all_float_probs - all_quant_probs
    prob_mse = np.mean(prob_diff ** 2)
    prob_mae = np.mean(np.abs(prob_diff))
    prob_max_err = np.max(np.abs(prob_diff))

    # --- KL divergence per sample, then average ---
    # D_KL(P || Q) = sum(P * log(P / Q))
    eps = 1e-12
    kl_div = np.sum(
        all_float_probs * np.log((all_float_probs + eps) / (all_quant_probs + eps)),
        axis=1,
    )
    kl_mean = np.mean(kl_div)
    kl_max = np.max(kl_div)

    # --- Accuracy (as percentage) ---
    float_preds = np.argmax(all_float_logits, axis=1)
    quant_preds = np.argmax(all_quant_logits, axis=1)

    float_acc = np.mean(float_preds == all_labels) * 100.0
    quant_acc = np.mean(quant_preds == all_labels) * 100.0
    acc_drop = float_acc - quant_acc
    agreement = np.mean(float_preds == quant_preds) * 100.0

    # --- Per-class probability MSE ---
    num_classes = all_float_logits.shape[1]
    per_class_mse = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            c_mse = np.mean(prob_diff[mask] ** 2)
        else:
            c_mse = 0.0
        per_class_mse.append(c_mse)

    # --- Confidence change ---
    float_conf = np.max(all_float_probs, axis=1)
    quant_conf = np.max(all_quant_probs, axis=1)
    mean_conf_drop = np.mean(float_conf - quant_conf) * 100.0

    # --- Print report ---
    print()
    print("=" * 62)
    print("  QUANTIZATION LOSS REPORT")
    print("=" * 62)
    print(f"  Validation samples : {len(val_ds)}")
    print()
    print("  ┌─ Probability-level error (softmax outputs) ────┐")
    print(f"  │  Mean Squared Error (MSE) : {prob_mse:.6e}      │")
    print(f"  │  Mean Absolute Error (MAE): {prob_mae:.6e}      │")
    print(f"  │  Max Absolute Error       : {prob_max_err:.6e}  │")
    print(f"  │  Mean KL divergence       : {kl_mean:.6e}       │")
    print(f"  │  Max KL divergence        : {kl_max:.6e}        │")
    print(f"  │  Mean confidence drop     : {mean_conf_drop:.2f} % │")
    print("  └──────────────────────────────────────────────────┘")
    print()
    print("  ┌─ Accuracy (argmax) ────────────────────────────┐")
    print(f"  │  Float accuracy           : {float_acc:6.2f} %   │")
    print(f"  │  Quantized accuracy       : {quant_acc:6.2f} %   │")
    print(f"  │  Accuracy drop            : {acc_drop:6.2f} %   │")
    print(f"  │  Prediction agreement     : {agreement:6.2f} %   │")
    print("  └──────────────────────────────────────────────────┘")
    print()
    print("  ┌─ Per-class probability MSE ────────────────────┐")
    for c in range(num_classes):
        print(f"  │  Class {c:<2}  MSE = {per_class_mse[c]:.6e}            │")
    print("  └──────────────────────────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()

    # Repository root
    repo_root = Path(__file__).resolve().parent.parent

    slug = f"{args.dataset}-{args.model}"

    onnx_path = repo_root / "src" / "models" / f"{slug}.onnx"

    out_dir = repo_root / "esp-dl" / "main" / "model"

    espdl_path = out_dir / "model.espdl"

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------

    if not onnx_path.exists():
        print(
            f"ERROR: ONNX model not found: {onnx_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    input_shape = infer_input_shape(args.dataset)

    n_calib_samples = args.calib_steps * CALIB_BATCH

    print("=== ESP-PPQ quantisation ===")
    print(f"  Model      : {slug}")
    print(f"  ONNX       : {onnx_path}")
    print(f"  Output slug: model.espdl")
    print(f"  Target     : {TARGET} ({args.bits}-bit)")
    print(f"  Input shape: {input_shape} (batch excluded)")
    print(f"  Device     : {args.device}")

    # -----------------------------------------------------------------------
    # Calibration data
    # -----------------------------------------------------------------------

    print("\n[1/3] Building calibration dataloader ...")

    calib_loader = load_calibration(
        args.dataset,
        repo_root,
        n_calib_samples,
    )

    # -----------------------------------------------------------------------
    # Quantisation
    # -----------------------------------------------------------------------

    print("\n[2/3] Running PTQ ...")

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target=TARGET,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        device=args.device,
        export_test_values=True,
        error_report=True,
        skip_export=False,
        verbose=1,
        dispatching_override=None,
    )

    # -----------------------------------------------------------------------
    # Quantization loss evaluation on full validation set
    # -----------------------------------------------------------------------

    print("\n[3/4] Evaluating quantization loss on validation set ...")

    _, val_ds, _ = load_datasets(args.dataset)

    evaluate_quantization_loss(
        quant_graph=quant_graph,
        onnx_path=str(onnx_path),
        val_ds=val_ds,
        device=args.device,
    )

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    print("[4/4] Outputs written:")

    for suffix in (".espdl", ".info", ".json"):
        p = espdl_path.with_suffix(suffix)

        if p.exists():
            size_kb = p.stat().st_size / 1024

            print(f"  {p.relative_to(repo_root)} ({size_kb:.1f} KB)")

    print("\nDone. Flash with:\n  idf.py flash monitor -p /dev/ttyUSB0")


if __name__ == "__main__":
    main()
