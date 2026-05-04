import os

import optuna
import optunahub
from optuna.trial import TrialState
import torch
import torch.optim as optim
import torch.utils.data
import subprocess
import select
import re
import time
import pty
from multiprocessing import Pool, set_start_method

from .models import TinyMambaMulti, TinyMamba3Multi
from .data import get_data_input_size, get_data_output_size, load_har_data, load_speechcommands_data
from .train import train, test
from .onnx import export_onnx

from filelock import FileLock
_device_lock = FileLock("/tmp/mcu.lock")


DEVICE = torch.device("cuda")
BATCHSIZE = 128
EPOCHS = 20
dataset_dir = "./data"
MODEL = "mamba-1"
DATASET = "har"
MULTI_LAYER = False

if MODEL == "mamba-1":
    N_WORKERS = 1
else:
    N_WORKERS = 3

if MULTI_LAYER:
    STUDY_NAME = f"{MODEL}-{DATASET}-multi-layer"
else:
    STUDY_NAME = f"{MODEL}-{DATASET}"

STORAGE_URL = "sqlite:///mamba_hpo.db"


def parse_device_result(output: str) -> tuple[bool, float | None]:
    if "panicked at" in output:
        return False, None
    if "INFERENCE_OK" in output:
        m = re.search(r"Latency (\d+)", output)
        return True, int(m.group(1)) if m else None
    raise ValueError("Unexpected output from device. Check that MCU is connected")


def run_on_device(timeout: float = 120.0):
    with _device_lock:
        env = {
            **os.environ,
            "MODEL": "trial-model",
            "DATASET": DATASET,
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
    d_model = trial.suggest_int("d_model", 8, 32)
    d_state = trial.suggest_int("d_state", 8, 16)
    d_conv = trial.suggest_int("d_conv", 2, 4)
    expand = trial.suggest_int("expand", 1, 4)
    if MULTI_LAYER:
        n_layers = trial.suggest_int("n_layers", 1, 10)
    else:
        n_layers = 1

    model = TinyMambaMulti(
        input_dim=get_data_input_size(DATASET),
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        output_size=get_data_output_size(DATASET),
    )
    return model


def define_mamba3_model(trial):
    d_model = trial.suggest_int("d_model", 8, 32, step=4)
    d_state = trial.suggest_int("d_state", 8, 16, step=2)
    expand = trial.suggest_int("expand", 1, 4)

    if MULTI_LAYER:
        n_layers = trial.suggest_int("n_layers", 1, 10)
    else:
        n_layers = 1

    d_inner = d_model * expand
    nheads = trial.suggest_categorical("nheads", [1, 2, 4, 8])
    if d_inner % (2 * nheads) != 0:
        raise optuna.exceptions.TrialPruned()
    headdim = d_inner // nheads
    model = TinyMamba3Multi(
        get_data_input_size(DATASET),
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        expand=expand,
        n_layers=n_layers,
        output_size=get_data_output_size(DATASET),
    )
    return model


def objective(trial):
    # Generate the model.
    if MODEL == "mamba-1":
        model = define_mamba1_model(trial).to(DEVICE)
    elif MODEL == "mamba-3":
        model = define_mamba3_model(trial).to(DEVICE)
    else:
        raise ValueError("Unknown model type:", MODEL)


    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    if DATASET == "har":
        train_ds, valid_ds, _ = load_har_data(dataset_dir)
    else:
        train_ds, valid_ds, _ = load_speechcommands_data(dataset_dir)

    train_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True, "shuffle": True}
    validate_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_ds, **validate_kwargs)

    # Train one epoch for more consistent latency
    train(model, DEVICE, train_loader, optimizer, 0, print_stats=True)
    onnx_path = f"src/models/{DATASET}-trial-model.onnx"
    export_onnx(model, DATASET, onnx_path, DEVICE)

    # Early test on device to check if it fits in memory
    success, latency_ms = run_on_device()
    if not success:
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        print(f"Model didn't run successfully on the MCU ({params_str})")
        raise optuna.exceptions.TrialPruned()

    # Training of the model.
    for epoch in range(1, EPOCHS-2):
        train(model, DEVICE, train_loader, optimizer, epoch, print_stats=True)

    accuracy = test(model, DEVICE, valid_loader)
    return accuracy, latency_ms


def run_optimization(_):
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    )
    study.optimize(objective, n_trials=100 // N_WORKERS)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        directions=["maximize", "minimize"],
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        load_if_exists=True,
    )
    study.set_metric_names(["Accuracy", "Latency"])

    if N_WORKERS > 1:
        # Make multiprocessing work with cuda
        set_start_method("spawn")
        with Pool(processes=N_WORKERS) as pool:
            pool.map(run_optimization, range(N_WORKERS))
    else:
        run_optimization(0)

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
