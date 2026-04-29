import os

import torch
import torch.optim as optim
import torch.utils.data
import subprocess
import select
import re
import time
import pty
import numpy as np

from .models import TinyMamba, TinyMamba3
from .data import load_har_data
from .train import train, test
from .onnx import export_onnx


DEVICE = torch.device("cuda")
BATCHSIZE = 128
# CLASSES = 10
# DIR = os.getcwd()
EPOCHS = 20
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
dataset_dir = "./data"
MODEL = "mamba1"

STUDY_NAME = f"{MODEL}-har"
STORAGE_URL = "sqlite:///mamba_hpo.db"


def parse_device_result(output: str) -> tuple[bool, float | None]:
    if "panicked at" in output:
        return False, None
    if "INFERENCE_OK" in output:
        m = re.search(r"Latency (\d+)", output)
        return True, int(m.group(1)) if m else None
    raise ValueError("Unexpected output from device. Check that MCU is connected")


def run_on_device(timeout: float = 120.0):
    env = {
        **os.environ,
        "MODEL": "trial-model",
        "DATASET": "har",
    }

    cmd = ["cargo", "run", "--release", "--quiet"]

    # Open new pty to not let process interfere with current PTY
    master_fd, slave_fd = pty.openpty()

    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        text=False,
        close_fds=True,
    )

    os.close(slave_fd)

    start_time = time.time()
    buffer = b""

    try:
        while True:
            if (time.time() - start_time) > timeout:
                proc.terminate()
                return False, None

            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in rlist:
                data = os.read(master_fd, 1024)
                if not data:
                    break

                buffer += data

                # Optional: real-time parsing trigger
                if b"INFERENCE_OK" in buffer or b"panicked at" in buffer:
                    proc.terminate()
                    break

        proc.wait(timeout=5)

    finally:
        os.close(master_fd)

    output = buffer.decode(errors="ignore")

    return parse_device_result(output)


def define_mamba1_model():
    d_model = 8
    d_state = 12
    d_conv = 2
    expand = 2
    model = TinyMamba(57, d_model, d_state, d_conv, expand, 6)
    return model


# def define_mamba3_model(trial):
#     d_model = trial.suggest_int("d_model", 8, 32, step=4)
#     d_state = trial.suggest_int("d_state", 8, 16, step=2)
#     expand = 2
#     d_inner = d_model * expand
#     nheads  = trial.suggest_categorical("nheads", [1, 2, 4, 8])
#     if d_inner % nheads != 0:
#         raise optuna.exceptions.TrialPruned()
#     headdim = d_inner // nheads
#     model = TinyMamba3(
#         57,
#         d_model=d_model,
#         d_state=d_state,
#         headdim=headdim,
#         expand=expand,
#         output_size=6,
#     )
#     return model


def train_model_and_test_accuracy(seed):
    torch.manual_seed(seed)
    # Generate the model.
    if MODEL == "mamba1":
        model = define_mamba1_model().to(DEVICE)
    else:
        model = define_mamba3_model().to(DEVICE)

    # Generate the optimizers.
    lr = 0.0054943708267906795
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds, valid_ds, _ = load_har_data(dataset_dir)

    train_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True, "shuffle": True}
    validate_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_ds, **validate_kwargs)

    # Training of the model.
    for epoch in range(EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch)
        accuracy = test(model, DEVICE, valid_loader)

        # trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # onnx_path = "src/models/har-trial-model.onnx"
    # export_onnx(model, "har", onnx_path, DEVICE)
    #
    # success, latency_ms = run_on_device()      # flash + parse
    # if not success:
    #     params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
    #     print(f"Model didn't run successfully on the MCU ({params_str})")
    #     raise optuna.exceptions.TrialPruned()
    print(f"Finished run with accuracy {accuracy*100:.3f}%")
    return accuracy


if __name__ == "__main__":
    print("Measuring variance")
    print(f"Training for {EPOCHS} epochs")
    seeds = range(10)
    accuracies = [train_model_and_test_accuracy(seed=s) for s in seeds]

    sigma_total = np.std(accuracies, ddof=1)
    target_ci = 0.005
    target_sigma = target_ci / 1.96

    k_needed = int(np.ceil((sigma_total / target_sigma) ** 2))

    print(f"empirical std:              {sigma_total*100:.3f}%")
    print(f"required σ_avg for CI<0.5%: {target_sigma*100:.3f}%")
    print(f"runs needed:                {k_needed}")
