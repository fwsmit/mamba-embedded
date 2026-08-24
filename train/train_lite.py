"""
Train the 7 fixed-architecture Mamba-1 models used to compare latency with the
Mamba-Lite micro paper.

Mamba-Lite micro spec (every Mamba hyperparameter at its library default,
only d_model=64 customised, i.e. Mamba(d_model=64)):

    Dataset                   input_dim   seq_len   d_model   n_classes
    Speech Commands v2 (KWS)  40          100       64        10
    UCI-HAR (HAR)             57          10        64        6

The 7 variants are:
    HAR:  bidir(add), bidir(mul), single-direction
    KWS:  bidir(add), bidir(mul), single-direction, bidir(mul) 2-layer

Reuses the existing MambaWrapper, train/test routines and ONNX export kernel
patching. The KWS MFCCs are loaded from the repo's precomputed (.., 49, 40)
pickles, filtered to the 10 command-word classes, and time-resampled to the
100-frame sequence length used by Mamba-Lite micro. When no CUDA device is
present the CUDA-only Mamba kernels are patched out with the CPU reference
implementations from onnx_utils/mamba_cpu_funcs.

Since the purpose is MCU latency / memory comparison (not accuracy), only light
training is performed: a small number of epochs, on a subset for the large KWS
set.

Usage:
    conda run -n torch-pascal python -m train.train_lite [--epochs N] [--model NAME]
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset

from mamba_ssm import Mamba

from .models import MambaWrapper
from .data import load_har_data, load_speechcommands_data
from .train import train, test as test_accuracy

# Mamba-Lite micro comparison config: Mamba defaults, only d_model=64.
D_MODEL = 64

# Mamba library defaults (kept explicit so the model is well-defined):
D_STATE = 16
D_CONV = 4
EXPAND = 2

HAR_FEAT, HAR_SEQ, HAR_CLS = 57, 10, 6
KWS_FEAT, KWS_SEQ, KWS_CLS = 40, 100, 10

# (name, dataset, bidirectional, strategy, n_layers)
SPECS = [
    ("har-bidir-add",   "har", True,  "add", 1),
    ("har-bidir-mul",   "har", True,  "ew_multiply", 1),
    ("har-single",      "har", False, None, 1),
    ("kws-bidir-add",   "kws", True,  "add", 1),
    ("kws-bidir-mul",   "kws", True,  "ew_multiply", 1),
    ("kws-single",      "kws", False, None, 1),
    ("kws-bidir-mul-2", "kws", True,  "ew_multiply", 2),
]

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.001
DEFAULT_GAMMA = 0.7

# KWS training is capped to a small subset (the full training set is 726k
# samples, far too slow / memory-hungry on CPU at seq_len=100 and unnecessary
# for a latency comparison).
KWS_MAX_TRAIN = 512
KWS_MAX_VAL = 256
KWS_THREADS = 4


def _patch_cpu_kernels():
    """Redirect CUDA-only Mamba kernels to CPU reference implementations."""
    import mamba_ssm.modules.mamba_simple as _mamba_mod
    import mamba_ssm.ops.triton.layernorm_gated as _ln_mod
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
    from .mamba_cpu_funcs import _selective_scan_vectorized

    _mamba_mod.causal_conv1d_fn = None
    _mamba_mod.selective_scan_fn = _selective_scan_vectorized
    _ln_mod.rmsnorm_fn = rms_norm_ref


def _load_lite_data(dataset, dataset_dir):
    """Return (train_ds, val_ds) with Mamba-Lite micro shapes.

    HAR is used as-is (10x57, 6 classes). KWS is filtered to the 10
    command-word classes and time-resampled 49 -> KWS_SEQ frames.
    """
    if dataset == "har":
        train_ds = load_har_data(dataset_dir, split="train")
        val_ds = load_har_data(dataset_dir, split="val")
        return train_ds, val_ds

    # KWS: precomputed pickles are (.., 49, 40) with 12 classes.
    train_ds = load_speechcommands_data(dataset_dir, split="train")
    val_ds = load_speechcommands_data(dataset_dir, split="val")

    def prep(ds, max_n):
        X, y = ds.X, ds.y
        keep = y < KWS_CLS                    # 10 command words (silence/unknown excluded)
        X, y = X[keep], y[keep]
        X = X[:max_n]
        y = y[:max_n]
        Xt = X.unsqueeze(1)                   # (N, 1, 49, 40)
        Xt = F.interpolate(Xt, size=(KWS_SEQ, KWS_FEAT),
                           mode="bilinear", align_corners=False)
        Xt = Xt.squeeze(1)                    # (N, 100, 40)
        return TensorDataset(Xt, y)

    return prep(train_ds, KWS_MAX_TRAIN), prep(val_ds, KWS_MAX_VAL)


def _export_onnx(model, seq_len, feat_dim, out_path, device):
    """Export a model with an explicit sequence length (Mamba-Lite shapes)."""
    from .onnx_utils import replace_unexportable_functions, put_back_unexportable_functions

    model.eval()
    dummy = torch.randn(1, seq_len, feat_dim, device=device)
    replace_unexportable_functions()
    try:
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["input"], output_names=["output"],
            verbose=False, opset_version=18, external_data=False,
            optimize=True, dynamo=False, do_constant_folding=True,
        )
    finally:
        put_back_unexportable_functions()


def train_one(spec, out_dir, epochs, batch_size, lr, gamma, device):
    name, dataset, bidir, strategy, n_layers = spec
    if dataset == "har":
        input_dim, seq_len, output_dim = HAR_FEAT, HAR_SEQ, HAR_CLS
    else:
        input_dim, seq_len, output_dim = KWS_FEAT, KWS_SEQ, KWS_CLS

    torch.manual_seed(0)
    model = MambaWrapper(
        mamba_model=Mamba,
        n_layers=n_layers,
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        bidirectional=bidir,
        bidirectional_strategy=strategy,
    ).to(device)

    train_ds, val_ds = _load_lite_data(dataset, os.path.expanduser("~/Datasets"))

    # drop_last: the CPU reference selective_scan only supports full batches.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, print_stats=True)
        test_accuracy(model, device, val_loader, print_stats=True)
        scheduler.step()

    onnx_path = os.path.join(out_dir, f"{name}.onnx")
    _export_onnx(model, seq_len, input_dim, onnx_path, device)
    print(f"[{name}] exported -> {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-Lite micro comparison models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(os.getcwd(), "mambalite-micro"))
    parser.add_argument("--model", type=str, default=None,
                        help="Only train this spec name (e.g. har-bidir-mul)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        _patch_cpu_kernels()
        # Keep the CPU selective_scan memory footprint bounded when running
        # the large-seq-len KWS models.
        torch.set_num_threads(KWS_THREADS)
        print(f"Patched Mamba kernels with CPU references (no CUDA); threads={torch.get_num_threads()}")

    os.makedirs(args.out_dir, exist_ok=True)

    specs = SPECS
    if args.model:
        specs = [s for s in specs if s[0] == args.model]
        if not specs:
            raise SystemExit(f"Unknown model {args.model!r}")

    for spec in specs:
        print(f"\n=== Training {spec[0]} (n_layers={spec[4]}, bidir={spec[2]}, "
              f"strategy={spec[3]}) ===")
        train_one(spec, args.out_dir, args.epochs, args.batch_size,
                  DEFAULT_LR, DEFAULT_GAMMA, device)

    print("\nAll models exported to:", args.out_dir)


if __name__ == "__main__":
    main()
