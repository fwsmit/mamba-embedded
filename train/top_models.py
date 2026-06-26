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
    quantize_onnx_to_espdl,
    load_calibration,
    load_datasets,
    evaluate_quantization_loss,
    get_input_quantization,
    quantize_dataset_to_bin,
    infer_input_shape,
    collate_fn,
    CALIB_STEPS,
    CALIB_BATCH,
    TARGET,
    NUM_OF_BITS,
)


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
    Extract the average single-inference latency (in microseconds) from the
    firmware's stdout.

    The firmware outputs a line like:
        I (1037) mamba_har: Average single-inference latency: 8172.1 us (8.172 ms).

    Returns the latency in microseconds (e.g., 8172.1), or None if not found.
    """
    import re

    match = re.search(
        r"Average single-inference latency:\s*([\d.]+)\s*us",
        stdout,
    )
    if match:
        return float(match.group(1))
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
    "count" (int) and "total_latency_us" (int), or None if the table
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
            "total_latency_us": total_latency,
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
) -> tuple[DataLoader, torch.utils.data.Dataset, np.ndarray]:
    """Return calibration loader, validation dataset, and validation labels."""
    print("Building calibration dataloader ...")
    calib_loader = load_calibration(dataset, repo_root, n_calib_samples)
    print(f"  Loaded {n_calib_samples} calibration samples")

    print("Building validation dataloader ...")
    val_ds = load_datasets(dataset, split="val")
    print(f"  Loaded {len(val_ds)} validation samples")
    print()

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    val_labels = np.concatenate([target.numpy() for _, target in val_loader])

    return calib_loader, val_ds, val_labels


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


def quantize_trial(
    trial_number: int,
    study_name: str,
    onnx_dir: Path,
    experiments_dir: Path,
    calib_loader: DataLoader,
    val_ds: torch.utils.data.Dataset,
    device: str,
    results: list[dict],
) -> None:
    """Quantize one trial's ONNX model and append its metrics to *results*."""
    src_onnx = onnx_dir / f"{study_name}-trial-{trial_number}.onnx"
    if not src_onnx.exists():
        print(f"  WARNING: ONNX file not found, skipping trial #{trial_number}: {src_onnx}")
        return

    onnx_path = experiments_dir / src_onnx.name
    shutil.copy2(src_onnx, onnx_path)

    espdl_path = onnx_path.with_suffix(".espdl")
    input_shape = infer_input_shape(onnx_path)

    print(f"  Trial #{trial_number}: {src_onnx.name}")
    print(f"    Copy        : {onnx_path}")
    print(f"    Input shape : {input_shape}")
    print(f"    Output      : {espdl_path}")

    quant_graph = quantize_onnx_to_espdl(
        onnx_path=onnx_path,
        espdl_path=espdl_path,
        calib_loader=calib_loader,
        calib_steps=CALIB_STEPS,
        input_shape=input_shape,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=device,
        collate_fn=collate_fn,
    )

    print(f"    Exporting quantized validation dataset ...")
    configs = get_input_quantization(quant_graph)
    dataset_bin_path = experiments_dir / f"dataset-trial-{trial_number}.bin"
    quantize_dataset_to_bin(configs, val_ds, dataset_bin_path)

    print(f"    Evaluating on validation set ...")
    metrics = evaluate_quantization_loss(
        quant_graph=quant_graph,
        onnx_path=str(onnx_path),
        val_ds=val_ds,
        device=device,
    )
    metrics.pop("accuracy_drop", None)
    metrics["trial_number"] = trial_number
    results.append(metrics)

    results_path = experiments_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Done.\n")


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
                    entry.pop("accuracy_drop", None)
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
    latency_us = parse_latency(output_text)
    if latency_us is not None:
        for entry in results:
            if entry["trial_number"] == trial_number:
                entry["mcu_latency_us"] = float(round(latency_us, 1))
                break
        results_path = experiments_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    MCU latency: {latency_us:.1f} us ({latency_us / 1000:.3f} ms)")
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
) -> None:
    """Run the full pipeline (select, quantize, deploy) for one study."""
    # Load study and select top models
    complete, values, trial_numbers = load_complete_trials(study_name, args.storage)
    selected_trials = select_top_models(values, trial_numbers, args.top_fraction)

    # Infer dataset and set up directories
    dataset = infer_dataset(study_name)
    onnx_dir = Path.home() / "Models" / study_name
    experiments_dir = repo_root / "experiments" / study_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Build data loaders (reused for all models in this study)
    calib_loader, val_ds, val_labels = build_data(dataset, repo_root, n_calib_samples)

    # Load previously quantized results to avoid rework
    results, done_trials = load_existing_results(experiments_dir)

    # Quantize each selected model
    print("=" * 62)
    print(f"  QUANTIZING SELECTED MODELS — {study_name}")
    print("=" * 62)
    print()

    for tn in selected_trials:
        if tn in done_trials:
            print(f"  Trial #{tn}: already quantized, skipping\n")
            continue
        quantize_trial(tn, study_name, onnx_dir, experiments_dir,
                       calib_loader, val_ds, device, results)

    # Final persist
    results_path = experiments_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")
    print("All selected models quantized and evaluated.")

    # Deploy each quantized model to ESP32-S3
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
        study_name = study_name_from_config(config_path)
        print()
        print("#" * 62)
        print(f"#  Processing study: {study_name}  (from {config_path})")
        print("#" * 62)
        print()

        process_study(study_name, args, repo_root, device, n_calib_samples, run_script)


if __name__ == "__main__":
    main()
