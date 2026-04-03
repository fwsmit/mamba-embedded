import pandas as pd
import torch
import torch.nn.functional as F
import os


def load_har_data(data_dir):
    def load_txt(file_path):
        return pd.read_csv(file_path, sep=r"\s+", header=None).values

    X_train = load_txt(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = load_txt(os.path.join(data_dir, "train", "y_train.txt")).squeeze() - 1
    X_test = load_txt(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = load_txt(os.path.join(data_dir, "test", "y_test.txt")).squeeze() - 1

    def prepare(X):
        X = F.pad(torch.tensor(X, dtype=torch.float32), (0, 570 - 561))
        return X.view(-1, 10, 57)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test
