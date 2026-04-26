import os

import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
import torch.utils.data
import subprocess
import select
import re
import time
import pty

from .models import TinyMamba, TinyMamba3
from .data import load_har_data, load_mnist_data, load_speechcommands_data
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
MODEL = "mamba3"

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


def define_mamba1_model(trial):
    d_model = trial.suggest_int("d_model", 8, 24, step=4)
    d_state = trial.suggest_int("d_state", 8, 16, step=2)
    d_conv = trial.suggest_int("d_conv", 2, 4)
    expand = trial.suggest_int("expand", 1, 4)
    model = TinyMamba(57, d_model, d_state, d_conv, expand, 6)
    return model


def define_mamba3_model(trial):
    d_model = trial.suggest_int("d_model", 8, 32, step=4)
    d_state = trial.suggest_int("d_state", 8, 16, step=2)
    expand = 2
    d_inner = d_model * expand
    nheads  = trial.suggest_categorical("nheads", [1, 2, 4, 8])
    if d_inner % nheads != 0:
        raise optuna.exceptions.TrialPruned()
    headdim = d_inner // nheads
    model = TinyMamba3(
        57,
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        expand=expand,
        output_size=6,
    )
    return model


def objective(trial):
    # Generate the model.
    if MODEL == "mamba1":
        model = define_mamba1_model(trial).to(DEVICE)
    else:
        model = define_mamba3_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_ds, valid_ds, _ = load_har_data(dataset_dir)

    train_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True, "shuffle": True}
    validate_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_ds, **validate_kwargs)

    MAX_PARAMS = 3 * 2 * 64**2  # assume max hidden dim = 64
    # size_ratio = MAX_PARAMS / model.approx_params()
    # alpha = 0.25  # memory pressure parameter

    # Training of the model.
    for epoch in range(EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch)
        accuracy = test(model, DEVICE, valid_loader)

        # trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    onnx_path = "src/models/har-trial-model.onnx"
    export_onnx(model, "har", onnx_path, DEVICE)

    success, latency_ms = run_on_device()      # flash + parse
    if not success:
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        print(f"Model didn't run successfully on the MCU ({params_str})")
        raise optuna.exceptions.TrialPruned()
    return accuracy, latency_ms

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(),
        load_if_exists=True,
    )
    study.set_metric_names(["Accuracy", "Latency"])
    study.optimize(objective, n_trials=30, timeout=3600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n── Study statistics ──────────────────────────")
    print(f"  Finished trials : {len(study.trials)}")
    print(f"  Pruned trials   : {len(pruned_trials)}")
    print(f"  Complete trials : {len(complete_trials)}")

    print("\n── Best trials ────────────────────────────────")
    for trial in study.best_trials:
        print("\n── Trial ────────────────────────────────")
        print(f"  Accuracy : {trial.values[0]:.6f}")
        print(f"  Latency   : {trial.values[1]:.6f}")   # e.g. second objective
        print("  Params   :")
        for key, value in trial.params.items():
            print(f"    {key:<12} = {value}")
