import pandas as pd
import torch
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
