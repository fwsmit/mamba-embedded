#!/usr/bin/env python3
"""
Select the top fraction of models from an Optuna HPO study using greedy
hypervolume contribution selection, then quantize each selected model to
ESP-DL .espdl format.

Usage:
    python -m train.top_models --study-name mamba-1-kws-2
    python -m train.top_models --study-name mamba-1-har
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

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
                # p_j dominates p_i ?
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
    # Sort by first coordinate ascending
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
    t[:, 0] = -values[:, 0]   # maximise → minimise
    t[:, 1] = values[:, 1]    # already minimise
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
# CLI
# ---------------------------------------------------------------------------


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

    # First line is num_samples; skip it.
    predictions = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            predictions.append(int(line))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top fraction of models from an Optuna study "
            "by greedy hypervolume contribution."
        )
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Optuna study name (e.g. mamba-1-kws-2)",
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load study -------------------------------------------------------
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    complete = [
        t for t in study.trials if t.state == TrialState.COMPLETE
    ]
    if not complete:
        print("ERROR: no complete trials found in study.", file=sys.stderr)
        sys.exit(1)

    values = [(t.values[0], t.values[1]) for t in complete]
    trial_numbers = [t.number for t in complete]

    print(f"Study           : {args.study_name}")
    print(f"Complete trials : {len(complete)}")
    print()

    # ---- Reference point --------------------------------------------------
    ref = make_reference_point(values)
    print(f"Reference point : acc ≤ {ref[0]:.4f}  lat ≥ {ref[1]:.1f} µs")

    # ---- Greedy selection ------------------------------------------------
    n_select = max(1, int(len(values) * args.top_fraction))
    print(f"Selecting top   : {n_select} of {len(values)} trials "
          f"(top {args.top_fraction * 100:.0f}%)")
    print()

    sel = greedy_hypervolume_selection(values, n_select, ref)

    selected_trials = [trial_numbers[i] for i in sel]
    print(f"\nSelected trial numbers: {sorted(selected_trials)}")

    # ---- (Optional) Write output file ------------------------------------
    if args.output:
        with open(args.output, "w") as f:
            for tn in selected_trials:
                f.write(f"{tn}\n")
        print(f"Written to {args.output}")

    # ---- Quantize each selected model -----------------------------------
    print()
    print("=" * 62)
    print("  QUANTIZING SELECTED MODELS")
    print("=" * 62)
    print()

    # Infer dataset from study name (second component, e.g. "mamba-1-kws-2" -> "kws")
    study_parts = args.study_name.split("-")
    known_datasets = {"har", "kws"}
    dataset = None
    for part in study_parts:
        if part in known_datasets:
            dataset = part
            break
    if dataset is None:
        print(
            f"ERROR: could not infer dataset from study name '{args.study_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    onnx_dir = Path.home() / "Models" / args.study_name

    # Copy ONNX models to experiments/STUDY_NAME/ so originals are never modified
    experiments_dir = repo_root / "experiments" / args.study_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    n_calib_samples = CALIB_STEPS * CALIB_BATCH

    print(f"  Dataset          : {dataset}")
    print(f"  ONNX directory   : {onnx_dir}")
    print(f"  Experiments dir  : {experiments_dir}")
    print(f"  Calibration steps: {CALIB_STEPS}")
    print()

    # Build calibration and validation dataloaders, reused for all models
    print("Building calibration dataloader ...")
    calib_loader = load_calibration(dataset, repo_root, n_calib_samples)
    print(f"  Loaded {n_calib_samples} calibration samples")

    print("Building validation dataloader ...")
    val_ds = load_datasets(dataset, split="val")
    print(f"  Loaded {len(val_ds)} validation samples")
    print()

    # Load existing results to avoid re-quantizing already-done trials
    import json as _json
    results_path = experiments_dir / "results.json"
    results: list = []
    if results_path.exists():
        with open(results_path, "r") as _f:
            results = _json.load(_f)
        done_trials = {r["trial_number"] for r in results}
        print(f"  Loaded {len(results)} existing results from {results_path}")
    else:
        done_trials = set()

    for tn in selected_trials:
        if tn in done_trials:
            print(f"  Trial #{tn}: already quantized, skipping")
            continue
        src_onnx = onnx_dir / f"{args.study_name}-trial-{tn}.onnx"
        if not src_onnx.exists():
            print(f"  WARNING: ONNX file not found, skipping trial #{tn}: {src_onnx}")
            continue

        # Copy to experiments dir so original is never modified
        onnx_path = experiments_dir / src_onnx.name
        shutil.copy2(src_onnx, onnx_path)

        espdl_path = onnx_path.with_suffix(".espdl")

        input_shape = infer_input_shape(onnx_path)

        print(f"  Trial #{tn}: {src_onnx.name}")
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

        # Export quantized validation dataset for on-device inference
        print(f"    Exporting quantized validation dataset ...")
        configs = get_input_quantization(quant_graph)
        dataset_bin_path = experiments_dir / f"dataset-trial-{tn}.bin"
        quantize_dataset_to_bin(configs, val_ds, dataset_bin_path)

        # Evaluate quantized model against the ONNX baseline
        print(f"    Evaluating on validation set ...")
        metrics = evaluate_quantization_loss(
            quant_graph=quant_graph,
            onnx_path=str(onnx_path),
            val_ds=val_ds,
            device=device,
        )
        metrics.pop("accuracy_drop", None)  # Computed later as float_accuracy - mcu_accuracy
        metrics["trial_number"] = tn
        results.append(metrics)

        # Persist after every trial so partial results survive a crash
        with open(results_path, "w") as f:
            _json.dump(results, f, indent=2)

        print(f"    Done.")
        print()

    # ---- Final write (redundant but harmless) ------------------------------
    with open(results_path, "w") as f:
        _json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")

    print("All selected models quantized and evaluated.")

    # ---- Deploy each quantized model to ESP32-S3 ---------------------------
    print()
    print("=" * 62)
    print("  DEPLOYING TO ESP32-S3")
    print("=" * 62)
    print()

    run_script = repo_root / "run-esp.sh"

    # Collect validation ground truth labels for MCU accuracy computation
    from torch.utils.data import DataLoader as _DataLoader
    val_loader = _DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    val_labels = np.concatenate([target.numpy() for _, target in val_loader])

    for tn in selected_trials:
        src_espdl = experiments_dir / f"{args.study_name}-trial-{tn}.espdl"
        if not src_espdl.exists():
            print(f"  WARNING: .espdl file not found, skipping trial #{tn}: {src_espdl}")
            continue

        print(f"  Trial #{tn}: {src_espdl.name}")
        print(f"    Running run-esp.sh ...")
        result = subprocess.run(
            [str(run_script), str(src_espdl)],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        print(f"    Exit code: {result.returncode}")
        if result.stdout:
            print(f"    stdout:\n{result.stdout}")
        if result.stderr:
            print(f"    stderr:\n{result.stderr}")

        # Parse machine-readable predictions from firmware output
        predictions = parse_predictions(result.stdout)

        # Store output alongside the .espdl file
        output_path = src_espdl.with_suffix(".output")
        with open(output_path, "w") as f:
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"stdout:\n{result.stdout}")
            f.write(f"stderr:\n{result.stderr}")
        print(f"    Output saved: {output_path}")

        # Save predictions alongside results.json
        if predictions:
            import json as _json
            preds_path = experiments_dir / "predictions.json"
            preds: dict = {}
            if preds_path.exists():
                with open(preds_path, "r") as _f:
                    preds = _json.load(_f)
            preds[str(tn)] = predictions
            with open(preds_path, "w") as _f:
                _json.dump(preds, _f, indent=2)
            print(f"    Predictions saved for trial #{tn}: {len(predictions)} samples")

            # Compute MCU accuracy and update results entry
            if len(predictions) == len(val_labels):
                mcu_acc = np.mean(np.array(predictions) == val_labels) * 100.0
                for entry in results:
                    if entry["trial_number"] == tn:
                        entry.pop("accuracy_drop", None)
                        entry["mcu_accuracy"] = float(round(mcu_acc, 2))
                        break
                with open(results_path, "w") as _f:
                    _json.dump(results, _f, indent=2)
                print(f"    MCU accuracy: {mcu_acc:.2f} %")
            else:
                print(f"    WARNING: prediction count ({len(predictions)}) doesn't"
                      f" match validation samples ({len(val_labels)}),"
                      f" cannot compute MCU accuracy")
        else:
            print(f"    WARNING: no machine-readable predictions found in output")
        print()

    print("All selected models deployed.")


if __name__ == "__main__":
    main()