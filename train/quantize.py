from esp_ppq.api import export_ppq_graph
from esp_ppq.IR import BaseGraph, Operation
# from esp_ppq.passes import QuantizationOptimizationPass
from esp_ppq.core import TargetPlatform
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import Subset
from .data import load_har_data, load_speechcommands_data
from code import InteractiveConsole


from esp_ppq.api import espdl_quantize_onnx
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


def build_calibration_dataset(dataset: str, root: Path):
    """
    Returns the training dataset used for PTQ calibration.
    """

    data_dir = root / "data"

    if dataset == "har":
        train_ds, _, _ = load_har_data(data_dir)

    elif dataset == "kws":
        train_ds, _, _ = load_speechcommands_data(
            data_dir=data_dir,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_ds


def infer_input_shape(dataset: str):

    if dataset == "har":
        # HAR ONNX input:
        # [batch, time, features]
        return [1, 10, 57]

    elif dataset == "kws":
        # KWS ONNX input:
        # [batch, time, mfcc]
        return [1, 101, 40]

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

    train_ds = build_calibration_dataset(dataset, root)

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


def remove_spurious_conv1d_transposes(graph: BaseGraph) -> None:
    """
    esp-ppq inserts Transpose[0,2,1] → Conv → Transpose[0,2,1] around
    every nn.Conv1d. The ONNX Conv already expects NCL, so these are no-ops
    that corrupt the sequence-length dimension. This pass removes them.
    """
    to_remove = []

    for op in graph.operations.values():
        if op.type != "Conv":
            continue

        in_var = op.inputs[0]  # the tensor fed into Conv
        pre_op = in_var.source_op

        # Check if preceded by Transpose[0,2,1]
        if pre_op is None or pre_op.type != "Transpose":
            continue
        if list(pre_op.attributes.get("perm", [])) != [0, 2, 1]:
            continue

        # Check if followed by Transpose[0,2,1]
        out_var = op.outputs[0]
        if len(out_var.dest_ops) != 1:
            continue
        post_op = out_var.dest_ops[0]
        if post_op.type != "Transpose":
            continue
        if list(post_op.attributes.get("perm", [])) != [0, 2, 1]:
            continue

        print("Modifying conv operation")
        # The pre-Transpose input is the real Conv input
        real_input = pre_op.inputs[0]

        # Rewire: Conv takes real_input directly
        graph.remove_operation(pre_op, keep_coherence=True)

        # The post-Transpose output is what downstream nodes consume
        real_output = post_op.outputs[0]
        graph.remove_operation(post_op, keep_coherence=True)

        # Fix the output shape to match NCL (undo the seq-len inflation)
        out_var.shape = real_input.shape  # [1, 16, 10] not [1, 13, 16]

        to_remove.append(op.name)

    print(
        f"[conv1d patch] Removed {len(to_remove)} spurious Transpose pairs: {to_remove}"
    )

def dump_conv1d_neighbourhood(graph: BaseGraph):
    for op_name, op in graph.operations.items():
        if op.type != 'Conv':
            continue

        print(f"\n{'='*60}")
        print(f"Conv op: {op_name}")
        print(f"  op.type:       {op.type}")
        print(f"  op.attributes: {dict(op.attributes)}")

        print(f"\n  --- Inputs ({len(op.inputs)}) ---")
        for i, var in enumerate(op.inputs):
            src = var.source_op
            print(f"  [{i}] var.name={var.name!r}  shape={var.shape}")
            print(f"       source_op: {src.name!r} type={src.type!r}" if src else "       source_op: None (graph input / initializer)")
            if src:
                print(f"       source_op.attributes: {dict(src.attributes)}")

        print(f"\n  --- Outputs ({len(op.outputs)}) ---")
        for i, var in enumerate(op.outputs):
            print(f"  [{i}] var.name={var.name!r}  shape={var.shape}")
            for j, dst in enumerate(var.dest_ops):
                print(f"       dest_op[{j}]: {dst.name!r} type={dst.type!r}")
                print(f"       dest_op.attributes: {dict(dst.attributes)}")

def fix_gather_output_shapes(graph: BaseGraph) -> None:
    """
    esp-ppq incorrectly drops the index dimension for Gather nodes whose
    indices are a 1-element vector (shape=[1]). ONNX spec: output.shape =
    data.shape[:axis] + indices.shape + data.shape[axis+1:]
    This pass recomputes and corrects every Gather output shape.
    """
    fixed = 0
    for op in graph.operations.values():
        if op.type != 'Gather':
            continue

        data_var    = op.inputs[0]
        indices_var = op.inputs[1]
        out_var     = op.outputs[0]

        data_shape    = data_var.shape
        indices_shape = indices_var.shape
        stored_shape  = out_var.shape

        if data_shape is None or indices_shape is None:
            print(f"[gather patch] SKIP {op.name}: missing shape info")
            continue

        axis_raw = op.attributes.get('axis', 0)
        axis     = int(axis_raw) % len(data_shape)

        # ONNX Gather: scalar index (indices_shape=[]) drops the dim;
        # vector index (indices_shape=[N]) keeps it.
        if isinstance(indices_shape, (list, tuple)) and len(indices_shape) >= 1:
            correct = list(data_shape[:axis]) + list(indices_shape) + list(data_shape[axis+1:])
        else:
            # Scalar: drop the axis dim (this case is already handled correctly)
            correct = list(data_shape[:axis]) + list(data_shape[axis+1:])

        if correct != list(stored_shape or []):
            print(f"[gather patch] {op.name}: {stored_shape} → {correct}")
            out_var.shape = correct
            fixed += 1

    print(f"[gather patch] Fixed {fixed} Gather output shapes.")



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

    espdl_path = out_dir / f"{slug}.espdl"

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
    print(f"  Output     : {espdl_path}")
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

    # After quantize_torch_model(), before export:
    fix_gather_output_shapes(quant_graph)
    # dump_conv1d_neighbourhood(quant_graph)
    # remove_spurious_conv1d_transposes(quant_graph)
    #
    # # Then export as normal
    export_ppq_graph(
        graph=quant_graph,
        platform=TargetPlatform.ESPDL_INT8,
        graph_save_to="har-mamba-1.espdl",
        config_save_to="har-mamba-1.json",
    )

    # executor = TorchExecutor(
    #     graph=quant_graph,
    #     device="cpu",
    # )

    # ------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------

    # # HAR example input
    # x = torch.randn(1, 10, 57).float()
    #
    # # Forward pass
    # outputs = executor.forward(x)
    #
    # print(type(outputs))
    #
    # # PPQ usually returns list/dict depending on graph
    # if isinstance(outputs, list):
    #     y = outputs[0]
    # elif isinstance(outputs, dict):
    #     y = list(outputs.values())[0]
    # else:
    #     y = outputs
    #
    # print("Output shape:", y.shape)
    # print(y)
    #
    for op_str in quant_graph.operations:
        op = quant_graph.operations[op_str]
        if op.type == "Transpose" or op.type == "Conv":
            print(op)
            print("inputs", op.inputs)
            print("outputs", op.outputs)
            print("attributes", op.attributes)

    # console = InteractiveConsole(locals())
    # console.interact()
    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    print("\n[3/3] Outputs written:")

    for suffix in (".espdl", ".info", ".json"):
        p = espdl_path.with_suffix(suffix)

        if p.exists():
            size_kb = p.stat().st_size / 1024

            print(f"  {p.relative_to(repo_root)} ({size_kb:.1f} KB)")

    print("\nDone. Flash with:\n  idf.py flash monitor -p /dev/ttyUSB0")


if __name__ == "__main__":
    main()
