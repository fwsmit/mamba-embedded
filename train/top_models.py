#!/usr/bin/env python3
"""
Select the top fraction of models from an Optuna HPO study using greedy
hypervolume contribution selection.

Usage:
    python -m train.top_models --study-name mamba-1-kws-2
    python -m train.top_models --study-name mamba-1-har
"""

import argparse
import sys

import numpy as np
import optuna
from optuna.trial import TrialState


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


if __name__ == "__main__":
    main()