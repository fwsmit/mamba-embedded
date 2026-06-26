from esp_ppq.api import export_ppq_graph
from esp_ppq.IR import BaseGraph, Operation
# from esp_ppq.passes import QuantizationOptimizationPass
from esp_ppq.core import TargetPlatform
import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
import torch
from torch.utils.data import Subset
from .data import load_har_data, load_speechcommands_data
from code import InteractiveConsole

from esp_ppq.executor import TorchExecutor
from esp_ppq.IR import QuantableOperation
from esp_ppq.core import (
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)

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
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the ONNX model file",
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
        "--skip-loss-report",
        action="store_true",
        help="Skip loading the dataset and reporting quantization loss / model sizes",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_datasets(dataset: str, split: str = "train"):
    """
    Load a specific split of a dataset.

    Args:
        dataset: "har" or "kws"
        split: "train", "val", or "test"

    Returns the requested dataset split.
    """
    data_dir = Path.home() / "Datasets"

    if dataset == "har":
        return load_har_data(data_dir, split=split)

    elif dataset == "kws":
        return load_speechcommands_data(data_dir=str(data_dir), split=split)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


INPUT_SHAPE_MAP = {
    (1, 10, 57): "har",
    (1, 49, 40): "kws",
}


def infer_dataset_from_onnx(onnx_path: Path) -> str:
    """
    Load an ONNX model and infer the dataset from its input shape.

    Returns the dataset name (e.g. 'har', 'kws').
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    if not graph.input:
        raise ValueError(f"ONNX model has no inputs: {onnx_path}")

    inp = graph.input[0]
    shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)

    if shape not in INPUT_SHAPE_MAP:
        raise ValueError(
            f"Unknown input shape {shape} in {onnx_path}. "
            f"Known shapes: {list(INPUT_SHAPE_MAP.keys())}"
        )

    return INPUT_SHAPE_MAP[shape]


def infer_model_name(onnx_path: Path) -> str:
    """
    Infer the model architecture name from the ONNX filename.

    Expects filenames like '<dataset>-<model>.onnx', e.g. 'har-mamba-1.onnx'.
    Strips the dataset prefix and .onnx suffix to return e.g. 'mamba-1'.
    """
    stem = onnx_path.stem  # e.g. 'har-mamba-1'
    # Remove known dataset prefixes
    for prefix in INPUT_SHAPE_MAP.values():
        if stem.startswith(prefix + "-"):
            return stem[len(prefix) + 1:]
    # Fallback: return the whole stem
    return stem


def infer_input_shape(onnx_path: Path):
    """
    Load an ONNX model and return its input shape as a list.
    """
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    return [d.dim_value for d in inp.type.tensor_type.shape.dim]


# ---------------------------------------------------------------------------
# Calibration dataloader
# ---------------------------------------------------------------------------


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    x, _ = batch
    return x.to(DEVICE)


def load_calibration(
    dataset: str,
    root: Path,
    n_samples: int,
):
    """
    Build calibration dataloader from existing datasets.
    """

    train_ds = load_datasets(dataset, split="train")

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
    device: str = DEVICE,
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

    return {
        "float_accuracy": float(round(float_acc, 2)),
        "quantized_accuracy": float(round(quant_acc, 2)),
        "accuracy_drop": float(round(acc_drop, 2)),
        "prediction_agreement": float(round(agreement, 2)),
        "prob_mse": float(prob_mse),
        "prob_mae": float(prob_mae),
        "prob_max_err": float(prob_max_err),
        "kl_mean": float(kl_mean),
        "kl_max": float(kl_max),
        "mean_conf_drop": float(round(mean_conf_drop, 2)),
        "per_class_mse": [float(m) for m in per_class_mse],
    }


# ---------------------------------------------------------------------------
# Model size reporting
# ---------------------------------------------------------------------------


def _get_elem_size(data_type: int) -> int:
    """Return element size in bytes for an ONNX tensor data type."""
    type_size = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT64: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.BOOL: 1,
    }
    return type_size.get(data_type, 4)


def get_espdl_param_size(info_path: Path) -> int:
    """
    Compute the total parameter size (in bytes) from an ESP-DL .info file.

    Parses the ``initializers (...)`` section and sums element counts times
    their data-type sizes.

    Returns 0 if the file does not exist or cannot be parsed.
    """
    if not info_path.exists():
        return 0

    info_text = info_path.read_text()
    init_start = info_text.find("initializers (")
    if init_start == -1:
        return 0

    init_end = info_text.find(")", init_start)
    init_section = info_text[init_start:init_end]

    dtype_size = {
        "INT8": 1, "UINT8": 1,
        "INT16": 2, "UINT16": 2,
        "INT32": 4, "UINT32": 4,
        "INT64": 8, "UINT64": 8,
        "FLOAT": 4, "FLOAT16": 2, "BFLOAT16": 2,
        "DOUBLE": 8,
        "BOOL": 1,
    }

    param_size = 0
    for line in init_section.split("\n"):
        line = line.strip()
        if not line or line == "initializers (":
            continue
        if line.startswith("%"):
            line = line[1:]
        bracket = line.find("[")
        if bracket == -1:
            continue
        rest = line[bracket + 1:-1]
        parts = rest.split(", ", 1)
        if len(parts) != 2:
            continue
        dtype = parts[0]
        shape_str = parts[1]
        if shape_str == "scalar":
            num_elements = 1
        else:
            dims = [int(d) for d in shape_str.split("x")]
            num_elements = 1
            for d in dims:
                num_elements *= d
        param_size += num_elements * dtype_size.get(dtype, 1)

    return param_size


def report_model_sizes(onnx_path: Path, espdl_path: Path, repo_root: Path):
    """
    Report model file sizes, split by parameters and graph overhead.

    For the ONNX model, parameter size is computed from the initializers (raw
    element count times element data-type size). Graph overhead is the remainder
    of the file size (protobuf encoding of graph structure, node definitions,
    etc.).

    For the ESP-DL model (.espdl), the EDL2 format is:
      char[4]  magic ("EDL2")
      uint32   mode (0 = no encryption)
      uint32   flatbuffer data size
      uint32   zero padding
      uint8[]  flatbuffer data

    Parameter size is computed from the .info file by summing all initializer
    element counts times their data-type sizes. Graph overhead is the flatbuffer
    data size minus the parameter size.
    """
    print()
    print("  ┌─ Model sizes ──────────────────────────────────┐")

    # --- Original ONNX model ---
    onnx_size = onnx_path.stat().st_size

    model = onnx.load(str(onnx_path))

    param_size = 0
    for init in model.graph.initializer:
        num_elements = 1
        for d in init.dims:
            num_elements *= d
        param_size += num_elements * _get_elem_size(init.data_type)

    graph_overhead = onnx_size - param_size

    rel = onnx_path.relative_to(repo_root) if onnx_path.is_relative_to(repo_root) else onnx_path
    print(f"  │  ONNX model ({rel})        │")
    print(f"  │    Total      : {onnx_size / 1024:>8.1f} KB              │")
    print(f"  │    Parameters : {param_size / 1024:>8.1f} KB              │")
    print(f"  │    Graph      : {graph_overhead / 1024:>8.1f} KB              │")

    # --- Quantized ESP-DL model ---
    espdl_file = espdl_path.with_suffix(".espdl")
    info_file = espdl_path.with_suffix(".info")

    if espdl_file.exists():
        with open(espdl_file, "rb") as f:
            raw = f.read()

        total = len(raw)
        magic = raw[:4]
        flatbuf_size = struct.unpack("<I", raw[8:12])[0]
        header_overhead = 16  # 4 magic + 4 mode + 4 size + 4 padding
        trailing_pad = total - header_overhead - flatbuf_size

        espdl_param_size = get_espdl_param_size(info_file)

        espdl_graph_overhead = flatbuf_size - espdl_param_size

        reduction_pct = (param_size - espdl_param_size) / param_size * 100.0 if param_size > 0 else 0.0

        print(f"  │  ESP-DL model (model.espdl)                    │")
        print(f"  │    Total      : {total / 1024:>8.1f} KB              │")
        print(f"  │    Parameters : {espdl_param_size / 1024:>8.1f} KB              │")
        print(f"  │      ({reduction_pct:.1f}% reduction from float)        │")
        print(f"  │    Graph      : {espdl_graph_overhead / 1024:>8.1f} KB              │")
        if trailing_pad > 0:
            print(f"  │    Padding    : {trailing_pad / 1024:>8.1f} KB              │")

    print("  └──────────────────────────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Quantization function
# ---------------------------------------------------------------------------


def quantize_onnx_to_espdl(
    onnx_path: str | Path,
    espdl_path: str | Path,
    calib_loader: DataLoader,
    calib_steps: int = CALIB_STEPS,
    input_shape: list[int] | None = None,
    target: str = TARGET,
    num_of_bits: int = NUM_OF_BITS,
    device: str = DEVICE,
    collate_fn=collate_fn,
) -> BaseGraph:
    """
    Quantize an ONNX model to ESP-DL .espdl format.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the input ONNX model file.
    espdl_path : str or Path
        Path for the output .espdl file (directory is created if needed).
    calib_loader : DataLoader
        DataLoader supplying calibration samples.
    calib_steps : int
        Number of calibration batches to run.
    input_shape : list[int] or None
        Expected input shape. If None, inferred from the ONNX model.
    target : str
        Target platform (default: "esp32s3").
    num_of_bits : int
        Quantization bit-width (8 or 16).
    device : str
        Device for PPQ calibration: "cpu" or "cuda".
    collate_fn : callable
        Collate function for the calibration dataloader.

    Returns
    -------
    BaseGraph
        The quantized PPQ graph for further inspection or analysis.
    """
    onnx_path = Path(onnx_path)
    espdl_path = Path(espdl_path)

    if input_shape is None:
        input_shape = infer_input_shape(onnx_path)

    espdl_path.parent.mkdir(parents=True, exist_ok=True)

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=input_shape,
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=collate_fn,
        device=device,
        export_test_values=True,
        error_report=True,
        skip_export=False,
        verbose=0,
        dispatching_override=None,
    )

    return quant_graph


# ---------------------------------------------------------------------------
# Input quantization inspection
# ---------------------------------------------------------------------------


def get_input_quantization(
    graph: BaseGraph,
) -> dict[str, TensorQuantizationConfig]:
    """
    Extract the input quantization configuration from a quantized BaseGraph.

    Returns a dict mapping each graph-level input variable name to its
    ``TensorQuantizationConfig``, which describes how that tensor was quantized
    (scale, offset, bit-width, min/max, policy).

    Parameters
    ----------
    graph : BaseGraph
        The quantized graph returned by ``espdl_quantize_onnx``
        (or ``quantize_onnx_to_espdl``).

    Returns
    -------
    dict[str, TensorQuantizationConfig]
        A dictionary of {input_variable_name: TensorQuantizationConfig}.
        If no quantized input is found (e.g., the first op is not quantized),
        the value is ``None``.
    """
    result: dict[str, TensorQuantizationConfig] = {}

    for var_name, var in graph.inputs.items():
        tqc = None

        for dest_op in var.dest_ops:
            if not isinstance(dest_op, QuantableOperation):
                continue

            # Find which input slot this variable occupies
            try:
                idx = dest_op.inputs.index(var)
            except ValueError:
                # Variable not found in this op's inputs — skip
                continue

            tqc = dest_op.config.input_quantization_config[idx]
            break

        result[var_name] = tqc

    return result


def format_tqc(name: str, tqc: TensorQuantizationConfig | None) -> str:
    if tqc is None:
        return f"{name}: not quantized (FP32)"

    def _fmt_tensor(t) -> str:
        if t is None:
            return "None"
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                return f"{t.item():.6g}"
            return f"[{', '.join(f'{v:.6g}' for v in t.flatten().tolist())}]"
        return str(t)

    state_name = tqc.state.name if isinstance(tqc.state, QuantizationStates) else str(tqc.state)
    policy_str = str(tqc.policy).replace("QuantizationProperty.", "")

    lines = [
        f"{name}:",
        f"  state      : {state_name}",
        f"  bits       : {tqc.num_of_bits}",
        f"  scale      : {_fmt_tensor(tqc.scale)}",
        f"  offset     : {_fmt_tensor(tqc.offset)}",
        f"  quant_min  : {tqc.quant_min}",
        f"  quant_max  : {tqc.quant_max}",
        f"  policy     : {policy_str}",
    ]
    return "\n".join(lines)


def print_input_quant_configs(configs: dict[str, TensorQuantizationConfig | None]) -> None:
    for name, tqc in configs.items():
        print(format_tqc(name, tqc))
        print()


# ---------------------------------------------------------------------------
# Dataset quantization to binary file
# ---------------------------------------------------------------------------


def quantize_dataset_to_bin(
    configs: dict[str, TensorQuantizationConfig | None],
    dataset,
    output_path: str | Path,
) -> None:
    """
    Quantize a validation dataset using the input quantization config and write
    it as a flat binary file for inference on-device.

    The TQC defines the PPQ linear-quantization formula:
        q = clip(round(x / scale - offset), quant_min, quant_max)

    The binary layout (little-endian) is:
        [0..4)     uint32  number_of_samples
        [4..8)     uint32  elements_per_sample
        [8..)      int8[]  quantized data (flat, row-major, all samples
                           concatenated)

    Parameters
    ----------
    configs : dict[str, TensorQuantizationConfig | None]
        A dict mapping input variable names to their TQC (typically the output
        of ``get_input_quantization``). Only the first valid config is used.
    dataset : Dataset
        A PyTorch Dataset yielding ``(features, label)`` tuples where
        ``features`` is a float tensor of shape ``(..., C)``. The quantized
        output will be flattened.
    output_path : str or Path
        Path for the binary output file.
    """
    # Find the first valid (non-None) TQC
    tqc = None
    for _name, config in configs.items():
        if config is not None:
            tqc = config
            break

    if tqc is None:
        print("WARNING: No quantized input config found — skipping dataset quantization.")
        return

    scale = tqc.scale.cpu()
    offset = tqc.offset.cpu()
    quant_min = tqc.quant_min
    quant_max = tqc.quant_max

    print(f"  Quantizing {len(dataset)} validation samples with TQC:")
    print(f"    scale={scale.item() if scale.numel() == 1 else 'per-channel'}, "
          f"offset={offset.item() if offset.numel() == 1 else 'per-channel'}, "
          f"bits={tqc.num_of_bits}, range=[{quant_min}, {quant_max}]")

    all_quant = []
    elements_per_sample = None

    for i, (features, _label) in enumerate(dataset):
        # features: (seq_len, feat_dim) — add batch dim so shape matches
        # what the quantizer saw during calibration
        if features.dim() == 2:
            x = features.unsqueeze(0)  # (1, seq_len, feat_dim)
        else:
            x = features  # already has batch dim

        # PPQ quantisation formula
        unscaled = x / scale - offset
        q = torch.clamp(torch.round(unscaled), quant_min, quant_max).to(torch.int8)

        flat = q.flatten()
        if elements_per_sample is None:
            elements_per_sample = flat.numel()
        all_quant.append(flat)

    num_samples = len(all_quant)
    assert elements_per_sample is not None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Number of samples (uint32)
        f.write(struct.pack("<I", num_samples))
        # Elements per sample (uint32)
        f.write(struct.pack("<I", elements_per_sample))
        # Concatenated quantized data
        for q in all_quant:
            f.write(q.numpy().tobytes())

    size_kb = output_path.stat().st_size / 1024
    print(f"  Wrote {num_samples} samples × {elements_per_sample} int8 values → {output_path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()

    # Repository root
    repo_root = Path(__file__).resolve().parent.parent

    onnx_path = args.onnx_path.resolve()

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------

    if not onnx_path.exists():
        print(
            f"ERROR: ONNX model not found: {onnx_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Infer dataset and model name from the ONNX file
    dataset = infer_dataset_from_onnx(onnx_path)
    model_name = infer_model_name(onnx_path)

    slug = f"{dataset}-{model_name}"

    input_shape = infer_input_shape(onnx_path)

    out_dir = repo_root / "esp-dl" / "main" / "model"

    espdl_path = out_dir / "model.espdl"

    out_dir.mkdir(parents=True, exist_ok=True)

    n_calib_samples = args.calib_steps * CALIB_BATCH

    print("=== ESP-PPQ quantisation ===")
    print(f"  Model      : {slug}")
    print(f"  Dataset    : {dataset}")
    print(f"  ONNX       : {onnx_path}")
    print(f"  Output slug: model.espdl")
    print(f"  Target     : {TARGET} ({args.bits}-bit)")
    print(f"  Input shape: {input_shape}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device     : {device}")

    # -----------------------------------------------------------------------
    # Calibration data
    # -----------------------------------------------------------------------

    print("\n[1/4] Building calibration dataloader ...")

    calib_loader = load_calibration(
        dataset,
        repo_root,
        n_calib_samples,
    )

    # -----------------------------------------------------------------------
    # Quantisation
    # -----------------------------------------------------------------------

    print("\n[2/4] Running PTQ ...")

    quant_graph = quantize_onnx_to_espdl(
        onnx_path=onnx_path,
        espdl_path=espdl_path,
        calib_loader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=input_shape,
        target=TARGET,
        num_of_bits=args.bits,
        device=device,
        collate_fn=collate_fn,
    )

    configs = get_input_quantization(quant_graph)
    print_input_quant_configs(configs)

    # -----------------------------------------------------------------------
    # Export quantized validation dataset
    # -----------------------------------------------------------------------

    print("\n[3/4] Exporting quantized validation dataset ...")
    val_ds = load_datasets(dataset, split="val")

    dataset_bin_path = out_dir / "dataset.bin"
    quantize_dataset_to_bin(configs, val_ds, dataset_bin_path)

    # -----------------------------------------------------------------------
    # Quantization loss evaluation on full validation set
    # -----------------------------------------------------------------------

    if not args.skip_loss_report:
        print("\n[3/4] Evaluating quantization loss on validation set ...")

        evaluate_quantization_loss(
            quant_graph=quant_graph,
            onnx_path=str(onnx_path),
            val_ds=val_ds,
            device=device,
        )
    else:
        print("\n[3/4] Skipping quantization loss evaluation (--skip-loss-report)")

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    print("[4/4] Outputs written:")

    report_model_sizes(onnx_path, espdl_path, repo_root)

    print("Done. Flash with:\n  idf.py flash monitor -p /dev/ttyUSB0")


if __name__ == "__main__":
    main()
