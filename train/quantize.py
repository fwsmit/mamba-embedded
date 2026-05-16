import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import Subset
from .data import load_har_data, load_speechcommands_data

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
        choices=["mamba-1", "mamba-1-fixed", "mamba-3"],
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
            n_mfcc=40,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_ds


def infer_input_shape(dataset: str):

    if dataset == "har":
        # HAR ONNX input:
        # [batch, time, features]
        return [10, 57]

    elif dataset == "kws":
        # KWS ONNX input:
        # [batch, time, mfcc]
        return [101, 40]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# ---------------------------------------------------------------------------
# Calibration dataloader
# ---------------------------------------------------------------------------

def collate_fn(batch):

    # Case 1:
    # batch already is a tensor
    if isinstance(batch, torch.Tensor):
        return batch.float()

    # Case 2:
    # default DataLoader collation:
    # [inputs, labels]
    if (
        isinstance(batch, (list, tuple))
        and len(batch) == 2
        and isinstance(batch[0], torch.Tensor)
    ):
        return batch[0].float()

    # Case 3:
    # raw uncollated samples:
    # [(x,y), (x,y), ...]
    xs = []

    for x, _ in batch:

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        xs.append(x.float())

    return torch.stack(xs, dim=0)


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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    # Repository root
    repo_root = Path(__file__).resolve().parent.parent

    slug = f"{args.dataset}-{args.model}"

    onnx_path = (
        repo_root / "src" / "models" / f"{slug}.onnx"
    )

    out_dir = (
        repo_root / "esp-dl" / "main" / "model"
    )

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
        input_shape=[1] + input_shape,
        target=TARGET,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        device=args.device,
        export_test_values=True,
    )

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    print("\n[3/3] Outputs written:")

    for suffix in (".espdl", ".info", ".json"):

        p = espdl_path.with_suffix(suffix)

        if p.exists():

            size_kb = p.stat().st_size / 1024

            print(
                f"  {p.relative_to(repo_root)} "
                f"({size_kb:.1f} KB)"
            )

    print(
        "\nDone. Flash with:\n"
        "  idf.py flash monitor -p /dev/ttyUSB0"
    )


if __name__ == "__main__":
    main()
