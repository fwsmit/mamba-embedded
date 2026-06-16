import os

import numpy as np
import onnxruntime as ort
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
from hydra import main as hydra_main
from omegaconf import DictConfig
from mamba_ssm import Mamba, Mamba3

from .models import MambaWrapper
from .data import get_data_input_size, get_data_output_size, load_har_data, load_speechcommands_data
from .train import train, test
from .onnx_utils import export_onnx

from filelock import FileLock
_device_lock = FileLock("/tmp/mcu.lock")


DEVICE = torch.device("cuda")
dataset_dir = os.path.expanduser("~/Datasets")

BATCHSIZE = None
EPOCHS = None
MODEL = None
DATASET = None
MULTI_LAYER = None
EXPERIMENT_NAME = None
STUDY_NAME = None
ONNX_DIR = None
N_WORKERS = None
SEARCH_SPACE = None

STORAGE_URL = "sqlite:///mamba_hpo.db"

_PC_WARMUP_RUNS = 10
_PC_MEASURE_RUNS = 50


def parse_device_result(output: str) -> tuple[bool, float | None]:
    if "panicked at" in output:
        return False, None
    if "INFERENCE_OK" in output:
        m = re.search(r"Latency (\d+)", output)
        return True, int(m.group(1)) if m else None
    raise ValueError("Unexpected output from device. Check that MCU is connected")


def run_on_pc(onnx_path: str) -> tuple[bool, float | None]:
    """
    Estimate inference latency on the PC CPU using ONNX Runtime configured to
    mimic single-core bare-metal execution as closely as possible.

    Configuration rationale:
      - intra/inter_op_num_threads=1  : single-threaded, like the ESP32-S3 LX7
      - ORT_DISABLE_ALL optimizations : suppress SIMD fused kernels absent in burn-ndarray
      - os.sched_setaffinity          : pin to core 0 to reduce scheduler noise
      - batch_size=1                  : MCU always runs one sample at a time
      - median over _PC_MEASURE_RUNS  : robust to cache-miss and preemption outliers
      - shape from ONNX session       : avoids hardcoded shape mismatches

    Returns (success, latency_us) to match the signature of run_on_device().
    Latency is in microseconds so callers and Optuna objectives need no changes.

    Note: with N_WORKERS > 1 all workers will pin to core 0. Either accept the
    added noise, or pass the worker index through and pin each worker to a
    distinct core.
    """
    # Pin this process to a single core (Linux only; no-op on other platforms)
    try:
        os.sched_setaffinity(0, {0})
    except AttributeError:
        pass  # Windows / macOS — best-effort

    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        # Disable all graph optimisations so ORT doesn't fuse ops into AVX kernels
        # that burn-ndarray won't use on the MCU.
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Infer input shape directly from the model — always batch-size 1
        model_input = session.get_inputs()[0]
        shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in model_input.shape)
        dummy_input = np.random.randn(*shape).astype(np.float32)
        feed = {model_input.name: dummy_input}

        # Warmup: prime the instruction cache and allocate any internal ORT buffers
        for _ in range(_PC_WARMUP_RUNS):
            session.run(None, feed)

        # Measurement
        latencies_us = []
        for _ in range(_PC_MEASURE_RUNS):
            t0 = time.perf_counter()
            session.run(None, feed)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000)

        median_us = float(np.median(latencies_us))
        return True, median_us

    except Exception as exc:
        print(f"[run_on_pc] ONNX Runtime error: {exc}")
        return False, None


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


def _suggest_from_space(trial, name, space):
    if "choices" in space:
        return trial.suggest_categorical(name, space["choices"])
    if "step" in space:
        return trial.suggest_int(name, space["low"], space["high"], step=space["step"])
    return trial.suggest_int(name, space["low"], space["high"])


def define_mamba1_model(trial, search_space):
    d_model = _suggest_from_space(trial, "d_model", search_space.d_model)
    d_state = _suggest_from_space(trial, "d_state", search_space.d_state)
    d_conv = _suggest_from_space(trial, "d_conv", search_space.d_conv)
    expand = _suggest_from_space(trial, "expand", search_space.expand)
    if MULTI_LAYER:
        n_layers = _suggest_from_space(trial, "n_layers", search_space.n_layers)
    else:
        n_layers = 1

    model = MambaWrapper(
        mamba_model=Mamba,
        input_dim=get_data_input_size(DATASET)[1],
        output_dim=get_data_output_size(DATASET),
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        bidirectional=True,
    )
    return model


def define_mamba3_model(trial, search_space):
    d_model = _suggest_from_space(trial, "d_model", search_space.d_model)
    d_state = _suggest_from_space(trial, "d_state", search_space.d_state)
    expand = _suggest_from_space(trial, "expand", search_space.expand)

    if MULTI_LAYER:
        n_layers = _suggest_from_space(trial, "n_layers", search_space.n_layers)
    else:
        n_layers = 1

    d_inner = d_model * expand
    nheads = _suggest_from_space(trial, "nheads", search_space.nheads)
    if d_inner % (2 * nheads) != 0:
        raise optuna.exceptions.TrialPruned()
    headdim = d_inner // nheads
    model = MambaWrapper(
        mamba_model=Mamba3,
        input_dim=get_data_input_size(DATASET)[1],
        output_dim=get_data_output_size(DATASET),
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        expand=expand,
        n_layers=n_layers,
    )
    return model


def objective(trial):
    # Generate the model.
    if MODEL == "mamba-1":
        model = define_mamba1_model(trial, SEARCH_SPACE).to(DEVICE)
    elif MODEL == "mamba-3":
        model = define_mamba3_model(trial, SEARCH_SPACE).to(DEVICE)
    else:
        raise ValueError("Unknown model type:", MODEL)


    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    if DATASET == "har":
        train_ds, valid_ds, _ = load_har_data(dataset_dir)
    else:
        train_ds, valid_ds, _ = load_speechcommands_data(dataset_dir)

    train_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True, "shuffle": True}
    validate_kwargs = {"batch_size": BATCHSIZE, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_ds, **validate_kwargs)

    # Training of the model.
    for epoch in range(1, EPOCHS+1):
        train(model, DEVICE, train_loader, optimizer, epoch, print_stats=True)

    onnx_path = f"{ONNX_DIR}/{STUDY_NAME}-trial-{trial.number}.onnx"
    export_onnx(model, DATASET, onnx_path, DEVICE)
    success, latency_us = run_on_pc(onnx_path)
    if not success:
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        print(f"[trial {trial.number}] PC latency estimation failed ({params_str})")
        raise optuna.exceptions.TrialPruned()

    accuracy = test(model, DEVICE, valid_loader)
    return accuracy, latency_us


def run_optimization(_):
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    )
    study.optimize(
        objective,
        callbacks=[optuna.MaxTrialsCallback(100, states=(optuna.TrialState.COMPLETE))]
    )


@hydra_main(config_path="../config", config_name="arch-mamba1-kws", version_base=None)
def main(cfg: DictConfig):
    global BATCHSIZE, EPOCHS, MODEL, DATASET, MULTI_LAYER, EXPERIMENT_NAME
    global STUDY_NAME, ONNX_DIR, N_WORKERS, SEARCH_SPACE
    BATCHSIZE = cfg.BATCHSIZE
    EPOCHS = cfg.EPOCHS
    MODEL = cfg.MODEL
    DATASET = cfg.DATASET
    MULTI_LAYER = cfg.MULTI_LAYER
    EXPERIMENT_NAME = cfg.EXPERIMENT_NAME
    SEARCH_SPACE = cfg.SEARCH_SPACE
    STUDY_NAME = f"{MODEL}-{DATASET}-{EXPERIMENT_NAME}"
    ONNX_DIR = os.path.join(os.path.expanduser("~/Models"), STUDY_NAME)
    N_WORKERS = 1 if MODEL == "mamba-1" else 3

    print(f"Loaded configuration: {cfg}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        directions=["maximize", "minimize"],
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        load_if_exists=True,
    )
    study.set_metric_names(["Accuracy", "Latency"])
    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"── Starting architecture search ─────────────────")
    print(f"  Model          : {MODEL}")
    print(f"  Dataset        : {DATASET}")
    print(f"  Experiment     : {EXPERIMENT_NAME}")
    print(f"  Batch size     : {BATCHSIZE}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Multi-layer    : {MULTI_LAYER}")
    print(f"  Workers        : {N_WORKERS}")
    print(f"  ONNX dir       : {ONNX_DIR}")
    print(f"  Storage        : {STORAGE_URL}")
    print(f"  Study name     : {STUDY_NAME}")
    print()

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


if __name__ == "__main__":
    main()
