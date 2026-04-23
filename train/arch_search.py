import os

import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
import torch.utils.data
import subprocess
import re
import time

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


def parse_device_result(output: str) -> tuple[bool, float | None]:
    if "INFERENCE_OK" in output:
        m = re.search(r"Latency (\d+)", output)
        return True, int(m.group(1)) if m else None
    return False, None


def run_on_device(onnx_path: str, timeout: float = 120.0) -> tuple[bool, float | None]:
    env = {
        **os.environ,
        "MODEL": "trial-model",
        "DATASET": "har",
    }
    proc = subprocess.Popen(
        ["cargo", "run", "--release", "--quiet"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        # cwd=cwd,
    )

    output_lines = []
    try:
        # readline() blocks until a line arrives, so we need a deadline
        deadline = time.monotonic() + timeout
        
        for line in proc.stdout:
            output_lines.append(line)
            # print(line)
            if any(s in line for s in ("INFERENCE_OK", "INFERENCE_OOM", "panicked at")):
                # print("program finished, killing")
                break
            if time.monotonic() > deadline:
                break
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise  # let it propagate to Optuna
    finally:
        proc.kill()
        proc.wait()   # reap the process so it doesn't zombie

    output = "".join(output_lines)
    return parse_device_result(output)


def define_mamba1_model(trial):
    d_model = trial.suggest_int("d_model", 8, 32, step=4)
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
    try:
        # Generate the model.
        model = define_mamba1_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the FashionMNIST dataset.
        # train_loader, valid_loader = get_mnist()
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

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        onnx_path = "src/models/har-trial-model.onnx"
        export_onnx(model, "har", onnx_path, DEVICE)

        success, latency_us = run_on_device(onnx_path)      # flash + parse
        if not success:
            print("Model didn't run succesfully on the MCU")
            raise optuna.TrialPruned()
        # metric = accuracy * size_ratio ** alpha
        metric = accuracy
        return metric
    except KeyboardInterrupt:
        study.stop()
        raise

if __name__ == "__main__":
    # success, latency = run_on_device("src/models/har-mamba-1.onnx")
    # print(f"Success {success}, latency {latency}")
    # exit(0)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
