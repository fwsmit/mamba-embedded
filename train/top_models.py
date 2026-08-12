#!/usr/bin/env python3
"""
Select the top fraction of models from one or more Optuna HPO studies using
greedy hypervolume contribution selection, then quantize each selected model
to ESP-DL .espdl format.

Usage:
    python -m train.top_models config/arch-mamba1-kws.yaml
    python -m train.top_models config/arch-mamba1-kws.yaml config/arch-mamba1-har.yaml
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf
from optuna.trial import TrialState
from torch.utils.data import DataLoader

from .quantize import (
    get_espdl_param_size,
    quantize_onnx_to_espdl,
    quantize_onnx_to_espdl_best,
    QuantizationDivergedError,
    load_calibration,
    load_calibration_stratified,
    load_datasets,
    run_espdl_test,
    run_onnx_test,
    get_input_quantization,
    quantize_dataset_to_bin,
    infer_input_shape,
    collate_fn,
    CALIB_STEPS,
    CALIB_BATCH,
    TARGET,
    BEST_CALIB_STEPS,
    run_espdl_test,
    run_onnx_test,
)


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

# Supported quantization "methods" select both the esp-ppq quantize routine
# and its calibration loader. The ``strat-kl-tqt`` method is the best
# performing fully-8-bit config found (TQT_FINAL_REPORT.md): KL calibration
# over a stratified calibration set followed by a graph-wide
# TrainedQuantizationThreshold pass. Each method's metrics are stored under
# a suffixed results key and the .espdl artefact gets the matching filename
# suffix, so methods can be compared for the same set of selected models.
QUANTIZATION_METHODS = {
    "standard": {
        "quantize": quantize_onnx_to_espdl,
        "calib_steps": CALIB_STEPS,
        "key_suffix": "",
    },
    "strat-kl-tqt": {
        "quantize": quantize_onnx_to_espdl_best,
        "calib_steps": BEST_CALIB_STEPS,
        "key_suffix": "_strat",
    },
}

DEFAULT_QUANTIZATION_METHOD = "standard"


# ---------------------------------------------------------------------------
# Hypervolume helpers  (2-D only: accuracy maximised, latency minimised)
# ---------------------------------------------------------------------------


def _pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of *points* (both axes minimised)."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and not dominated[j]:
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    dominated[i] = True
                    break
    return points[~dominated]


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """
    2-D hypervolume for a minimisation problem.

    *points* ― (N, 2) array, both axes to be minimised.
    *ref*    ― (2,) reference point (worse than all points on every axis).

    Uses the standard O(N log N) sweep algorithm.
    """
    if len(points) == 0:
        return 0.0

    pf = _pareto_front(points)
    order = np.argsort(pf[:, 0])
    pf = pf[order]

    hv = 0.0
    prev_x = float(ref[0])
    for i in range(len(pf) - 1, -1, -1):
        x_i = pf[i, 0]
        y_i = pf[i, 1]
        width = prev_x - x_i
        height = ref[1] - y_i
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x_i
    return hv


def _transform(values: np.ndarray) -> np.ndarray:
    """
    Transform from (accuracy ↑, latency ↓) to (both minimised).

    Accuracy is negated so that larger accuracy becomes smaller (better)
    in the transformed space.
    """
    t = np.empty_like(values)
    t[:, 0] = -values[:, 0]
    t[:, 1] = values[:, 1]
    return t


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------


def greedy_hypervolume_selection(
    values: list[tuple[float, float]],
    n_select: int,
    ref_point: tuple[float, float],
) -> list[int]:
    """
    Greedily select *n_select* indices from *values*.

    At each step the trial whose addition increases the Pareto hypervolume
    the most is chosen.
    """
    arr = np.array(values)
    t_arr = _transform(arr)
    t_ref = _transform(np.array([ref_point]))[0]

    selected: list[int] = []
    remaining = list(range(len(values)))

    hv_cache: dict[frozenset, float] = {}

    def set_hv(indices: frozenset) -> float:
        if indices in hv_cache:
            return hv_cache[indices]
        pts = t_arr[list(indices)]
        hv = hypervolume(pts, t_ref)
        hv_cache[indices] = hv
        return hv

    for step in range(n_select):
        best_idx: int | None = None
        best_contrib = -1.0

        for idx in remaining:
            candidate = frozenset(set(selected) | {idx})
            hv_new = set_hv(candidate)
            hv_old = set_hv(frozenset(selected)) if selected else 0.0
            contrib = hv_new - hv_old
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx

        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
        acc, lat = values[best_idx]
        print(f"  Trial #{best_idx:>3}: acc={acc:.4f}  lat={lat:>8.2f} µs  "
              f"contrib={best_contrib:.6f}")

    return selected


# ---------------------------------------------------------------------------
# Reference-point heuristics
# ---------------------------------------------------------------------------


def make_reference_point(values: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Build a reference point guaranteed to be worse than every trial.

    Accuracy is worsened below the minimum, latency worsened above the
    maximum.
    """
    accs = [v[0] for v in values]
    lats = [v[1] for v in values]
    return (min(accs) - 0.01, max(lats) + 10.0)


# ---------------------------------------------------------------------------
# Predictions parsing
# ---------------------------------------------------------------------------


def parse_predictions(stdout: str) -> list[int]:
    """
    Extract machine-readable predictions from the firmware's stdout.

    The firmware outputs:
        ===PREDICTIONS_START===
        <num_samples>
        <prediction_0>
        <prediction_1>
        ...
        ===PREDICTIONS_END===

    Returns a list of predictions (one per sample), or an empty list if the
    markers are not found.
    """
    start_marker = "===PREDICTIONS_START==="
    end_marker = "===PREDICTIONS_END==="

    start_idx = stdout.find(start_marker)
    if start_idx == -1:
        return []
    end_idx = stdout.find(end_marker, start_idx)
    if end_idx == -1:
        return []

    body = stdout[start_idx + len(start_marker):end_idx].strip()
    lines = body.splitlines()
    if not lines:
        return []

    predictions = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            predictions.append(int(line))
    return predictions


def parse_latency(stdout: str) -> float | None:
    """
    Extract the average single-inference latency (in milliseconds) from the
    firmware's stdout.

    The firmware outputs a line like:
        I (1037) mamba_har: Average single-inference latency: 8172.1 us (8.172 ms).

    Returns the latency in milliseconds (e.g., 8.172), or None if not found.
    """
    import re

    match = re.search(
        r"Average single-inference latency:\s*([\d.]+)\s*us",
        stdout,
    )
    if match:
        return float(match.group(1)) / 1000.0
    return None


def parse_profiling_table(stdout: str) -> dict[str, dict] | None:
    """
    Extract the grouped-by-type profiling table from the firmware's stdout.

    The firmware outputs a table like:
        +--------------------+-------+---------------+-------------+
        |                     Grouped by type                     |
        +--------------------+-------+---------------+-------------+
        | type               | count | total latency | avg latency |
        +--------------------+-------+---------------+-------------+
        | Mul                |    27 |        2596us |        96us |
        | MatMul             |     5 |        1869us |       373us |
        ...
        +--------------------+-------+---------------+-------------+

    Returns a dict mapping operator type names to dicts with
    "count" (int) and "total_latency_ms" (float), or None if the table
    is not found.
    """
    import re

    # Find the table header
    header_pattern = r"\|\s*type\s*\|\s*count\s*\|\s*total latency\s*\|\s*avg latency\s*\|"
    header_match = re.search(header_pattern, stdout)
    if not header_match:
        return None

    # Find the separator line after the header
    after_header = stdout[header_match.end() :]
    sep_match = re.search(r"\+-+" , after_header)
    if not sep_match:
        return None

    # Now parse each data row until the closing separator
    rows_start = header_match.end() + sep_match.end()
    rows_text = stdout[rows_start:]

    # Match rows like: | Mul                |    27 |        2596us |        96us |
    row_pattern = re.compile(
        r"\|\s+(\S[^|]*?)\s+\|\s+(\d+)\s+\|\s+(\d+)us\s+\|"
    )

    table: dict[str, dict] = {}
    for row_match in row_pattern.finditer(rows_text):
        op_type = row_match.group(1).strip()
        count = int(row_match.group(2))
        total_latency = int(row_match.group(3))
        table[op_type] = {
            "count": count,
            "total_latency_ms": round(total_latency / 1000.0, 3),
        }

    return table if table else None


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding Optuna study name."""
    cfg = OmegaConf.load(config_path)
    name = f"{cfg.MODEL}-{cfg.DATASET}"
    if cfg.get("EXPERIMENT_NAME"):
        name += f"-{cfg.EXPERIMENT_NAME}"
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top fraction of models from one or more Optuna studies "
            "by greedy hypervolume contribution."
        )
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to Hydra config YAML files (one or more)",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///mamba_hpo.db",
        help="Optuna storage URL (default: sqlite:///mamba_hpo.db)",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction of complete trials to select (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write selected trial numbers (one per line) to this file",
    )
    return parser.parse_args(argv)


def load_complete_trials(
    study_name: str, storage: str,
) -> tuple[list[optuna.Trial], list[tuple[float, float]], list[int]]:
    """Load study and return complete trials with their values and numbers."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete:
        print("ERROR: no complete trials found in study.", file=sys.stderr)
        sys.exit(1)

    values = [(t.values[0], t.values[1]) for t in complete]
    trial_numbers = [t.number for t in complete]

    print(f"Study           : {study_name}")
    print(f"Complete trials : {len(complete)}")
    print()

    return complete, values, trial_numbers


def update_result(
    experiments_dir: Path,
    trial_number: int,
    key: str,
    value,
) -> list[dict]:
    """Load results.json, set `key` = `value` for the given trial_number, save back.

    If no entry exists yet for trial_number, a new one is created.
    Returns the full updated results list.
    """

    print(f"Updating results {trial_number}; {key} = {value}")
    results_path = experiments_dir / "results.json"

    results: list[dict] = []
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)

    for r in results:
        if r["trial_number"] == trial_number:
            r[key] = value
            break
    else:
        results.append({"trial_number": trial_number, key: value})

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Updated trial {trial_number}: {key} = {value} ({results_path})")
    return results


def select_top_models(
    values: list[tuple[float, float]],
    trial_numbers: list[int],
    top_fraction: float,
) -> list[int]:
    """Apply greedy hypervolume selection and return chosen trial numbers."""
    ref = make_reference_point(values)
    print(f"Reference point : acc ≤ {ref[0]:.4f}  lat ≥ {ref[1]:.1f} µs")

    n_select = max(1, int(len(values) * top_fraction))
    print(f"Selecting top   : {n_select} of {len(values)} trials "
          f"(top {top_fraction * 100:.0f}%)")
    print()

    sel = greedy_hypervolume_selection(values, n_select, ref)
    selected_trials = [trial_numbers[i] for i in sel]
    print(f"\nSelected trial numbers: {sorted(selected_trials)}")

    return selected_trials


def write_selected_trials(selected_trials: list[int], output_path: str) -> None:
    with open(output_path, "w") as f:
        for tn in selected_trials:
            f.write(f"{tn}\n")
    print(f"Written to {output_path}")


def infer_dataset(study_name: str) -> str:
    """Extract dataset name from study-name parts (e.g. 'mamba-1-kws-2' -> 'kws')."""
    known_datasets = {"har", "kws"}
    for part in study_name.split("-"):
        if part in known_datasets:
            return part
    print(
        f"ERROR: could not infer dataset from study name '{study_name}'",
        file=sys.stderr,
    )
    sys.exit(1)


def build_data(
    dataset: str, repo_root: Path, n_calib_samples: int,
) -> tuple[DataLoader, torch.utils.data.Dataset, np.ndarray,
            torch.utils.data.Dataset, np.ndarray]:
    """Return calibration loader, validation dataset/labels, and test dataset/labels."""
    print("Building calibration dataloader ...")
    calib_loader = load_calibration(dataset, repo_root, n_calib_samples)
    print(f"  Loaded {n_calib_samples} calibration samples")

    print("Building validation dataloader ...")
    val_ds = load_datasets(dataset, split="val")
    print(f"  Loaded {len(val_ds)} validation samples")

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    val_labels = np.concatenate([target.numpy() for _, target in val_loader])

    print("Building test dataloader ...")
    test_ds = load_datasets(dataset, split="test")
    print(f"  Loaded {len(test_ds)} test samples")
    print()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)
    test_labels = np.concatenate([target.numpy() for _, target in test_loader])

    return calib_loader, val_ds, val_labels, test_ds, test_labels


def build_calib_loader_for_method(
    dataset: str,
    repo_root: Path,
    method: str,
    n_calib_samples: int,
    train_ds=None,
):
    """Build the calibration dataloader appropriate for *method*.

    ``strat-kl-tqt`` needs a class-balanced stratified selection, whereas the
    standard method uses uniform-random subsampling.  *train_ds*, when given,
    is reused so the training set is loaded only once.
    """
    if method == "strat-kl-tqt":
        return load_calibration_stratified(dataset, repo_root, train_ds=train_ds)
    return load_calibration(dataset, repo_root, n_calib_samples, train_ds=train_ds)


def combined_key_suffix(num_of_bits: int, method: str) -> str:
    """Filename / results key suffix for a precision × method combination.

    Backward compatible: the standard method at 8 bits uses no suffix.
    """
    method_suffix = QUANTIZATION_METHODS[method]["key_suffix"]
    precision_suffix = f"_int{num_of_bits}" if num_of_bits != 8 else ""
    return method_suffix + precision_suffix


def resolve_methods(cfg) -> list[str]:
    """Read the requested quantization methods from a config, defaulting to
    the standard method for backward compatibility."""
    raw = cfg.get("quantization_methods", DEFAULT_QUANTIZATION_METHOD)
    if isinstance(raw, str):
        raw = [raw]
    for m in raw:
        if m not in QUANTIZATION_METHODS:
            raise ValueError(
                f"Unknown quantization method '{m}'. "
                f"Available: {list(QUANTIZATION_METHODS)}"
            )
    return list(raw)


def load_existing_results(
    experiments_dir: Path,
) -> tuple[list[dict], set[int]]:
    """Load previously saved results and return (results, done_trial_numbers)."""
    results_path = experiments_dir / "results.json"
    results: list[dict] = []
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
        done_trials = {r["trial_number"] for r in results}
        print(f"  Loaded {len(results)} existing results from {results_path}")
    else:
        done_trials = set()
    return results, done_trials


def count_float_params(onnx_path: str) -> int:
    """Count the number of trainable parameters in an ONNX model.

    Sums elements across all float-dtype initializer tensors (weights + biases),
    excluding integer (int64/int32) initializers used purely as shape/index/axis
    constants (e.g. for Reshape, Slice, Split, Gather ops).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f":  # float16 / float32 / float64
            total += arr.size
    return total


def quantize_trial(
    trial_number: int,
    study_name: str,
    onnx_dir: Path,
    experiments_dir: Path,
    calib_loader: DataLoader,
    val_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    device: str,
    results: list[dict],
    num_of_bits: int = 8,
    key_suffix: str = "",
    quantize_func=quantize_onnx_to_espdl,
    calib_steps: int = CALIB_STEPS,
    quant_key: str = "quantized_accuracy",
    test_quant_key: str = "test_quantized_accuracy",
):  # -> None:
    """Quantize one trial's ONNX model and add its metrics to *results*.

    If *key_suffix* is non-empty (e.g. "_int16"), quantisation-specific metrics
    are stored under suffixed keys (e.g. ``quantized_accuracy_int16``) to avoid
    overwriting entries from a previous precision.  The ``float_accuracy`` key
    is shared as it is the float baseline for all precisions.

    If an entry for this ``trial_number`` already exists in *results*, the new
    metrics are merged into it (adding suffixed keys); otherwise a new entry is
    created.
    """
    src_onnx = onnx_dir / f"{study_name}-trial-{trial_number}.onnx"
    if not src_onnx.exists():
        print(f"  WARNING: ONNX file not found, skipping trial #{trial_number}: {src_onnx}")
        return None

    onnx_path = experiments_dir / src_onnx.name
    if not onnx_path.exists():
        shutil.copy2(src_onnx, onnx_path)

    espdl_path = onnx_path.with_stem(f"{onnx_path.stem}{key_suffix}").with_suffix(".espdl")

    # if espdl_path.exists():
    #     print("Trail is already quantized. Skipping...")
    #     return espdl_path

    input_shape = infer_input_shape(onnx_path)

    print(f"  Trial #{trial_number}: {src_onnx.name} ({num_of_bits}-bit)")
    print(f"    Input shape : {input_shape}")
    print(f"    Output      : {espdl_path}")

    try:
        quant_graph = quantize_func(
            onnx_path=onnx_path,
            espdl_path=espdl_path,
            calib_loader=calib_loader,
            calib_steps=calib_steps,
            input_shape=input_shape,
            target=TARGET,
            num_of_bits=num_of_bits,
            device=device,
            collate_fn=collate_fn,
        )
    except QuantizationDivergedError as e:
        print(f"  WARNING: {e} Skipping trial #{trial_number} for this method.")
        entry: dict = {"trial_number": trial_number, quant_key: 0, test_quant_key: 0}
        existing = next((e for e in results if e.get("trial_number") == trial_number), None)
        if existing is not None:
            existing.update(entry)
        else:
            results.append(entry)
        results_path = experiments_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    Stored 0 accuracy for diverged trial #{trial_number}")
        return None

    print(f"    Exporting quantized validation dataset ...")
    configs = get_input_quantization(quant_graph)
    dataset_bin_path = experiments_dir / f"dataset-trial-{trial_number}{key_suffix}.bin"
    quantize_dataset_to_bin(configs, val_ds, dataset_bin_path)


    # Build entry: float_accuracy is shared, quant-specific keys get suffix
    entry: dict = {}
    entry["trial_number"] = trial_number
    # entry["float_accuracy"] = metrics.pop("float_accuracy")
    # for k, v in metrics.items():
    #     entry[f"{k}{key_suffix}"] = v
    # entry["test_float_accuracy"] = test_metrics.pop("float_accuracy")
    # for k, v in test_metrics.items():
    #     entry[f"test_{k}{key_suffix}"] = v

    # Compute quantised parameter size from the .info file
    info_path = espdl_path.with_suffix(".info")
    info_key = f"param_size_bytes{key_suffix}" if key_suffix else "param_size_bytes"
    entry[info_key] = get_espdl_param_size(info_path)
    print(f"    Param size  : {entry[info_key]} bytes")

    # Merge into existing entry if one exists, otherwise append new
    existing = next((e for e in results if e.get("trial_number") == trial_number), None)
    if existing is not None:
        existing.update(entry)
        print(f"    Merged into existing entry for trial #{trial_number}")
    else:
        results.append(entry)
        print(f"    Created new entry for trial #{trial_number}")

    results_path = experiments_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Done.\n")

    return quant_graph


def deploy_trial(
    trial_number: int,
    study_name: str,
    experiments_dir: Path,
    run_script: Path,
    results: list[dict],
    val_labels: np.ndarray,
) -> None:
    """Run the quantized model on ESP32-S3 and parse its output predictions."""
    run_on_mcu(trial_number, study_name, experiments_dir, run_script)
    parse_mcu_output(trial_number, experiments_dir, results, val_labels)


def run_on_mcu(
    trial_number: int,
    study_name: str,
    experiments_dir: Path,
    run_script: Path,
) -> None:
    """Run the quantized model on ESP32-S3 and save the raw output.

    Skips execution if the .output file already exists (indicating a
    previous successful run). Parsing of the saved output is done
    separately by parse_mcu_output().
    """
    src_espdl = experiments_dir / f"{study_name}-trial-{trial_number}.espdl"
    if not src_espdl.exists():
        print(f"  WARNING: .espdl file not found, skipping trial #{trial_number}: {src_espdl}")
        return

    output_path = src_espdl.with_suffix(".output")
    if output_path.exists():
        print(f"  Trial #{trial_number}: output already exists at {output_path}, skipping MCU run\n")
        return

    print(f"  Trial #{trial_number}: {src_espdl.name}")
    print(f"    Running run-esp.sh ...")
    result = subprocess.run(
        [str(run_script), str(src_espdl)],
        capture_output=True,
        text=True,
        cwd=run_script.parent,
    )
    print(f"    Exit code: {result.returncode}")
    if result.stdout:
        print(f"    stdout:\n{result.stdout}")
    if result.stderr:
        print(f"    stderr:\n{result.stderr}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Exit code: {result.returncode}\n")
        f.write(f"stdout:\n{result.stdout}")
        f.write(f"stderr:\n{result.stderr}")
    print(f"    Output saved: {output_path}")
    print()


def parse_mcu_output(
    trial_number: int,
    experiments_dir: Path,
    results: list[dict],
    val_labels: np.ndarray,
) -> None:
    """Read the saved .output file from a previous MCU run, parse
    predictions and latency, and update *results* in-place.

    This runs every time (no caching) so that results.json always
    reflects the latest parsing logic.
    """
    output_candidates = list(experiments_dir.glob(f"*-trial-{trial_number}.output"))
    if not output_candidates:
        print(f"  WARNING: no .output file found for trial #{trial_number}, skipping parse")
        return
    output_path = output_candidates[0]

    print(f"  Trial #{trial_number}: parsing {output_path}")
    with open(output_path, "r") as f:
        output_text = f.read()

    predictions = parse_predictions(output_text)

    if predictions:
        preds_path = experiments_dir / f"predictions_trial_{trial_number}.json"
        with open(preds_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"    Predictions saved: {preds_path} ({len(predictions)} samples)")

        if len(predictions) == len(val_labels):
            mcu_acc = np.mean(np.array(predictions) == val_labels) * 100.0
            for entry in results:
                if entry["trial_number"] == trial_number:
                    entry["mcu_accuracy"] = float(round(mcu_acc, 2))
                    break
            results_path = experiments_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    MCU accuracy: {mcu_acc:.2f} %")
        else:
            print(f"    WARNING: prediction count ({len(predictions)}) doesn't"
                  f" match validation samples ({len(val_labels)}),"
                  f" cannot compute MCU accuracy")
    else:
        print(f"    WARNING: no machine-readable predictions found in output")

    print("Parsing latency")
    latency_ms = parse_latency(output_text)
    if latency_ms is not None:
        for entry in results:
            if entry["trial_number"] == trial_number:
                entry["mcu_latency_ms"] = float(round(latency_ms, 2))
                break
        results_path = experiments_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    MCU latency: {latency_ms:.2f} ms")
    else:
        print(f"    WARNING: latency line not found in firmware output")

    print("Parsing profiling table")
    profiling = parse_profiling_table(output_text)
    if profiling is not None:
        for entry in results:
            if entry["trial_number"] == trial_number:
                entry["mcu_profiling"] = profiling
                break
        results_path = experiments_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    Profiling table: {len(profiling)} operator types parsed")
    else:
        print(f"    WARNING: profiling table not found in firmware output")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_study(
    study_name: str,
    args: argparse.Namespace,
    repo_root: Path,
    device: str,
    n_calib_samples: int,
    run_script: Path,
    num_of_bits_list: list[int],
    cfg: dict,
) -> None:
    """Run the full pipeline (select, quantize, deploy) for one study.

    *num_of_bits_list* may contain one or more bit-widths (e.g. ``[8]`` or
    ``[8, 16]``).  Each precision is quantised separately; metrics for
    non-8-bit widths are stored under suffixed keys (e.g. ``quantized_accuracy_int16``).
    Accuracy on the test set is stored under keys prefixed with ``test_``
    (e.g. ``test_float_accuracy``, ``test_quantized_accuracy_int16``).
    """
    # Load study and select top models
    complete, values, trial_numbers = load_complete_trials(study_name, args.storage)
    selected_trials = select_top_models(values, trial_numbers, args.top_fraction)

    # Infer dataset and set up directories
    dataset = infer_dataset(study_name)
    onnx_dir = Path.home() / "Models" / study_name
    experiments_dir = repo_root / "experiments" / study_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Build data loaders (reused for all models in this study)
    calib_loader, val_ds, val_labels, test_ds, test_labels = build_data(
        dataset, repo_root, n_calib_samples)

    # Quantization methods to run (e.g. standard + strat-kl-tqt for comparison)
    methods = resolve_methods(cfg)
    # Load the (potentially multi-GB) training set once and derive every
    # method's calibration subset from it, then release it.  The loaders hold
    # only the tiny compact subsets, so the full training set is freed.
    train_ds = load_datasets(dataset, split="train")
    calib_loaders = {
        m: build_calib_loader_for_method(dataset, repo_root, m, n_calib_samples, train_ds)
        for m in methods
    }
    del train_ds

    # Quantize each selected model at each requested precision × method
    for num_of_bits in num_of_bits_list:
        for method in methods:
            if num_of_bits == 16 and method == "strat-kl-tqt":
                print("Skipping 16-bit TQT quantization")
                continue
            key_suffix = combined_key_suffix(num_of_bits, method)
            suffix_desc = f"{num_of_bits}-bit/{method}"
            quant_key = f"quantized_accuracy{key_suffix}"
            test_quant_key = f"test_quantized_accuracy{key_suffix}"
            quant_func = QUANTIZATION_METHODS[method]["quantize"]
            calib_steps = QUANTIZATION_METHODS[method]["calib_steps"]

            print()
            print("=" * 62)
            print(f"  QUANTIZING SELECTED MODELS — {suffix_desc} — {study_name}")
            print("=" * 62)
            print()

            results, _ = load_existing_results(experiments_dir)
            # Determine which trials already have results for this precision/method
            done_valid_for_precision = set()
            done_test_for_precision = set()
            for entry in results:
                tn = entry.get("trial_number")
                if tn is not None and quant_key in entry:
                    done_valid_for_precision.add(tn)
                if tn is not None and test_quant_key in entry:
                    done_test_for_precision.add(tn)

            for tn in selected_trials:
                if tn in done_valid_for_precision and tn in done_test_for_precision:
                    print(f"  Trial #{tn}: {suffix_desc} already tested on validation and test set, skipping\n")
                    continue

                results, _ = load_existing_results(experiments_dir)
                quant_graph = quantize_trial(
                    tn, study_name, onnx_dir, experiments_dir,
                    calib_loaders[method], val_ds, test_ds, device, results,
                    num_of_bits=num_of_bits, key_suffix=key_suffix,
                    quantize_func=quant_func, calib_steps=calib_steps,
                    quant_key=quant_key, test_quant_key=test_quant_key,
                )

                if not quant_graph:
                    print("WARNING: quantization did not work, probably missing ONNX file")
                    continue

                assert (quant_graph)

                if tn not in done_valid_for_precision:
                    print(f"    Evaluating on validation set ...")
                    accuracy = run_espdl_test(quant_graph, val_ds, 'cuda')
                    update_result(experiments_dir, tn, quant_key, accuracy)

                if tn not in done_test_for_precision:
                    print(f"    Evaluating on test set ...")
                    accuracy = run_espdl_test(quant_graph, test_ds, 'cuda')
                    update_result(experiments_dir, tn, test_quant_key, accuracy)

    print("All selected models quantized and evaluated.")

    results, _ = load_existing_results(experiments_dir)
    done_valid_for_float = set()
    done_test_for_float = set()
    for entry in results:
        tn = entry.get("trial_number")
        if tn is not None and "float_accuracy" in entry:
            done_valid_for_float.add(tn)
        if tn is not None and "test_float_accuracy" in entry:
            done_test_for_float.add(tn)

    print("Calculation ONNX model accuracy")
    # Calculate ONNX precision
    for tn in selected_trials:
        onnx_path = experiments_dir / f"{study_name}-trial-{tn}.onnx"
        if tn in done_valid_for_float and tn in done_test_for_float:
            print(f"  Trial #{tn} ONNX: already tested on validation and test set, skipping\n")
            continue

        if not tn in done_valid_for_float:
            print(f"    Evaluating on validation set ...")
            accuracy = run_onnx_test(str(onnx_path), val_ds)
            update_result(experiments_dir, tn, "float_accuracy", accuracy)

        if not tn in done_test_for_float:
            print(f"    Evaluating on test set ...")
            accuracy = run_onnx_test(str(onnx_path), test_ds)
            update_result(experiments_dir, tn, "test_float_accuracy", accuracy)

    print("Calculation ONNX model parameter count")
    # Count true parameters from each trial's float ONNX model (not tied to
    # quantization). Skipped when already stored in results.json.
    results, _ = load_existing_results(experiments_dir)
    done_nr_params = {r["trial_number"]
                      for r in results if "nr_parameters" in r}
    for tn in selected_trials:
        if tn in done_nr_params:
            print(f"    Trial #{tn}: nr_parameters already stored, skipping\n")
            continue
        onnx_path = experiments_dir / f"{study_name}-trial-{tn}.onnx"
        print(f"    Count nr_parameters for Trial #{tn} ...")
        nr_parameters = count_float_params(str(onnx_path))
        update_result(experiments_dir, tn, "nr_parameters", nr_parameters)


    # Deploy each quantized model to ESP32-S3 (only 8-bit for now)
    print()
    print("=" * 62)
    print(f"  RUNNING ON ESP32-S3 — {study_name}")
    print("=" * 62)
    print()

    for tn in selected_trials:
        run_on_mcu(tn, study_name, experiments_dir, run_script)

    print()
    print("=" * 62)
    print(f"  PARSING MCU OUTPUT — {study_name}")
    print("=" * 62)
    print()

    results, _ = load_existing_results(experiments_dir)
    for tn in selected_trials:
        parse_mcu_output(tn, experiments_dir, results, val_labels)

    print("All selected models processed.")



def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = Path(__file__).resolve().parent.parent
    n_calib_samples = CALIB_STEPS * CALIB_BATCH
    run_script = repo_root / "run-esp.sh"

    for config_path in args.configs:
        cfg = OmegaConf.load(config_path)
        study_name = study_name_from_config(config_path)
        raw = cfg.get("quantization_precision", 8)
        # Normalise to a list (backward compat: scalar -> single-element list)
        if type(raw) is int:
            num_of_bits_list = [raw]
        else:
            num_of_bits_list = list(raw)
        prec_desc = "+".join(str(b) for b in num_of_bits_list)
        print()
        print("#" * 62)
        print(f"#  Processing study: {study_name}  (from {config_path})")
        print(f"#  Quantization precision(s): {prec_desc}-bit")
        print("#" * 62)
        print()

        process_study(study_name, args, repo_root, device, n_calib_samples, run_script,
                       num_of_bits_list=num_of_bits_list, cfg=cfg)


if __name__ == "__main__":
    main()
