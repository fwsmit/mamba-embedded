import pandas as pd
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms
import os


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

    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [0.8, 0.2])
    test_ds = TensorDataset(X_test, y_test)
    return train_ds, val_ds, test_ds

# The 35 classes in the full Speech Commands v2 dataset
CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

TARGET_SAMPLE_RATE = 16_000
SAMPLE_LENGTH = 16_000  # 1 second at 16 kHz


def load_speechcommands_data(data_dir, n_mfcc=40):
    """
    Load Speech Commands v2 with MFCC preprocessing following the Keyword Mamba
    paper: each sample is returned as X ∈ ℝ^(T × F), a sequence of T time-frame
    patches each carrying F = n_mfcc coefficients, ready for linear projection.

    Args:
        data_dir:  Root directory where the dataset is stored / downloaded.
        n_mfcc:    Number of MFCC coefficients (= frequency dimension F).

    Returns:
        train_ds, val_ds, test_ds — wrapped datasets ready for DataLoader.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=n_mfcc,               # F — frequency dimension
        melkwargs={
            "n_fft": 400,            # 25 ms window at 16 kHz
            # "hop_length": 160,       # 10 ms hop  → T ≈ 101 frames per second
            "hop_length": 320,       # 20 ms hop  → T ≈ 51 frames per second
            "n_mels": 80,
        },
    )

    def preprocess(waveform, sample_rate):
        # 1. Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)(waveform)

        # 2. Pad or truncate to exactly 1 second
        length = waveform.shape[-1]
        if length < SAMPLE_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_LENGTH - length))
        else:
            waveform = waveform[..., :SAMPLE_LENGTH]

        # 3. Compute MFCC: (1, F, T)
        mfcc = mfcc_transform(waveform)

        # 4. Remove channel dim and transpose → (T, F)
        #    Each row is one patch X_n ∈ ℝ^F (f=F, t=1 per the paper)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (F, T) → (T, F)

        return mfcc

    class SpeechCommandsWrapper(torch.utils.data.Dataset):
        """Thin wrapper that preprocesses samples and encodes labels."""

        def __init__(self, subset):
            self._ds = SPEECHCOMMANDS(data_dir, download=True, subset=subset)

        def __len__(self):
            # return len(self._ds)

            # FIXME turn this back
            return int(len(self._ds)/100)

        def __getitem__(self, idx):
            waveform, sample_rate, label, *_ = self._ds[idx]
            mfcc = preprocess(waveform, sample_rate)   # (T, F)
            target = LABEL_TO_IDX[label]
            return mfcc, target

    train_ds = SpeechCommandsWrapper("training")
    val_ds   = SpeechCommandsWrapper("validation")
    test_ds  = SpeechCommandsWrapper("testing")

    return train_ds, val_ds, test_ds
