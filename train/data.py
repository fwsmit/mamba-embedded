import pandas as pd
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms
import os
import hashlib



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


def _preprocessing_hash(n_mfcc, n_fft, hop_length, n_mels, sample_rate):
    """Stable short hash of all preprocessing parameters — used as cache key."""
    key = f"mfcc{n_mfcc}_fft{n_fft}_hop{hop_length}_mels{n_mels}_sr{sample_rate}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def load_speechcommands_data(data_dir, n_mfcc=40, cache_dir=None):
    """
    Load Speech Commands v2 with MFCC preprocessing following the Keyword Mamba
    paper: each sample is returned as X ∈ ℝ^(T × F), a sequence of T time-frame
    patches each carrying F = n_mfcc coefficients, ready for linear projection.

    Preprocessed features are cached to disk on the first pass and reused on
    subsequent runs. The cache is keyed by a hash of all preprocessing parameters,
    so changing n_mfcc, hop_length, etc. automatically triggers a fresh cache.

    Args:
        data_dir:   Root directory where the dataset is stored / downloaded.
        n_mfcc:     Number of MFCC coefficients (= frequency dimension F).
        cache_dir:  Root directory for the feature cache.  Defaults to
                    <data_dir>/.mfcc_cache.
    Returns:
        train_ds, val_ds, test_ds — wrapped datasets ready for DataLoader.
    """
    N_FFT      = 400
    HOP_LENGTH = 320

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, ".mfcc_cache")

    param_tag = _preprocessing_hash(n_mfcc, N_FFT, HOP_LENGTH, 80, TARGET_SAMPLE_RATE)
    versioned_cache = os.path.join(cache_dir, param_tag)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft":       N_FFT,
            "hop_length":  HOP_LENGTH,
            "n_mels":      80,
        },
    )

    def preprocess(waveform, sample_rate):
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                sample_rate, TARGET_SAMPLE_RATE
            )(waveform)
        length = waveform.shape[-1]
        if length < SAMPLE_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_LENGTH - length))
        else:
            waveform = waveform[..., :SAMPLE_LENGTH]
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0).transpose(0, 1)  # (T, F)

    class SpeechCommandsWrapper(torch.utils.data.Dataset):
        def __init__(self, subset):
            self._ds        = SPEECHCOMMANDS(data_dir, download=True, subset=subset)
            self._cache_dir = os.path.join(versioned_cache, subset)
            self._mem: dict[int, tuple] = {}
            os.makedirs(self._cache_dir, exist_ok=True)

        def __len__(self):
            return len(self._ds)

        def _cache_path(self, idx):
            # _walker holds the full audio file path — use it as a stable,
            # order-independent key rather than the integer index.
            audio_path = self._ds._walker[idx]
            rel = os.path.relpath(audio_path, data_dir)
            # Flatten the relative path into a single filename.
            safe_name = rel.replace(os.sep, "__")
            return os.path.join(self._cache_dir, safe_name + ".pt")

        def __getitem__(self, idx):
            if idx in self._mem:
                return self._mem[idx]

            path = self._cache_path(idx)
            if os.path.exists(path):
                item =  torch.load(path, weights_only=True)
            else:
                waveform, sample_rate, label, *_ = self._ds[idx]
                mfcc   = preprocess(waveform, sample_rate)   # (T, F)
                target = LABEL_TO_IDX[label]
                item = (mfcc, target)

                # Atomic write: write to a temp file then rename to avoid
                # leaving partial .pt files if the process is interrupted.
                tmp_path = path + ".tmp"
                torch.save((mfcc, target), tmp_path)
                os.replace(tmp_path, path)

            self._mem[idx] = item
            return item

    train_ds = SpeechCommandsWrapper("training")
    val_ds   = SpeechCommandsWrapper("validation")
    test_ds  = SpeechCommandsWrapper("testing")
    return train_ds, val_ds, test_ds


def get_data_input_size(dataset):
    if dataset == "mnist":
        input_dim = 28
    elif dataset == "har":
        input_dim = 57
    elif dataset == "kws":
        input_dim = 40
    else:
        raise ValueError("Unknown dataset type", dataset)
    return input_dim


def get_data_output_size(dataset):
    if dataset == "mnist":
        output_dim = 10
    elif dataset == "har":
        output_dim = 6
    elif dataset == "kws":
        output_dim = 35
    else:
        raise ValueError("Unknown dataset type", dataset)
    return output_dim
