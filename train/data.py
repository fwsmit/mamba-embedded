import pandas as pd
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import hashlib
from pathlib import Path
import numpy as np
import pickle
from .kws_dataset_gen import IDX2LABEL, N_FRAMES, N_MFCC, LABEL2IDX



def load_mnist_data(data_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds, val_ds = random_split(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        [0.8, 0.2],
    )
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    return train_ds, val_ds, test_ds


def load_har_data(data_dir):
    def load_txt(file_path):
        return pd.read_csv(file_path, sep=r"\s+", header=None).values

    har_data_dir = os.path.join(data_dir, "har-uci-dataset", "UCI HAR Dataset")
    X_train = load_txt(os.path.join(har_data_dir, "train", "X_train.txt"))
    y_train = load_txt(os.path.join(har_data_dir, "train", "y_train.txt")).squeeze() - 1
    X_test = load_txt(os.path.join(har_data_dir, "test", "X_test.txt"))
    y_test = load_txt(os.path.join(har_data_dir, "test", "y_test.txt")).squeeze() - 1

    def prepare(X):
        X = F.pad(torch.tensor(X, dtype=torch.float32), (0, 570 - 561))
        return X.view(-1, 10, 57)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    test_ds = TensorDataset(X_test, y_test)
    return train_ds, val_ds, test_ds


class SpeechCommandsMFCC(Dataset):
    """
    Lightweight wrapper around pre-computed MFCC arrays.

    Each ``__getitem__`` returns ``(mfcc, label)`` where

    * ``mfcc``  – ``FloatTensor`` of shape **(49, 40)**
    * ``label`` – ``LongTensor`` scalar in *[0, NUM_CLASSES)*

    Serialisation
    -------------
    >>> ds = SpeechCommandsMFCC(X, y)
    >>> ds.save("kws_data/train.pkl")
    >>> ds = SpeechCommandsMFCC.load("kws_data/train.pkl")
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 3 and X.shape[1:] == (N_FRAMES, N_MFCC), \
            f"X must be (N, {N_FRAMES}, {N_MFCC}), got {X.shape}"
        self.X = torch.from_numpy(X)   # (N, 49, 40)  float32
        self.y = torch.from_numpy(y)   # (N,)          int64

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]

    # ── I/O ───────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"X": self.X.numpy(), "y": self.y.numpy(),
                 "label2idx": LABEL2IDX, "idx2label": IDX2LABEL},
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )
        size_mb = path.stat().st_size / 1_048_576
        print(f"    ✓ {len(self):>7,} samples  →  {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str | Path) -> "SpeechCommandsMFCC":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data["X"], data["y"])

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def class_counts(self) -> dict[str, int]:
        """Return a {label: count} dictionary."""
        counts: dict[str, int] = {}
        for idx in self.y.tolist():
            lbl = IDX2LABEL[idx]
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    def __repr__(self) -> str:
        return (
            f"SpeechCommandsMFCC("
            f"n={len(self)}, shape={tuple(self.X.shape[1:])}, "
            f"n_classes={self.y.unique().numel()})"
        )


def load_speechcommands_data(data_dir):
    dataset_dir = "speech_commands_v0.02_augmented"
    dir = data_dir + "/" + dataset_dir
    train_ds = SpeechCommandsMFCC.load(dir + "/train.pkl")
    val_ds = SpeechCommandsMFCC.load(dir + "/val.pkl")
    test_ds = SpeechCommandsMFCC.load(dir + "/test.pkl")
    return train_ds, val_ds, test_ds


def get_data_input_size(dataset):
    if dataset == "mnist":
        input_dim = [28, 28]
    elif dataset == "har":
        input_dim = [10, 57]
    elif dataset == "kws":
        input_dim = [49, 40]
    else:
        raise ValueError("Unknown dataset type", dataset)
    return input_dim


def get_data_output_size(dataset):
    if dataset == "mnist":
        output_dim = 10
    elif dataset == "har":
        output_dim = 6
    elif dataset == "kws":
        output_dim = 12
    else:
        raise ValueError("Unknown dataset type", dataset)
    return output_dim
