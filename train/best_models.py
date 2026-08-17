#!/usr/bin/env python3
"""
Select the n best models across one or more Optuna HPO studies ranked by
validation accuracy / a cost metric. The metric is chosen with --sort:
"param" (default) ranks by validation accuracy / nr_parameters, "latency"
ranks by validation accuracy / mcu_latency_ms (only trials with MCU latency
data are considered).

For each config the study name is derived (as in top_models.py / plot_arch_search.py)
and its experiments/<study>/results.json is read. All trials from all studies are
merged, ranked by the chosen metric (descending), and the top n are printed in a table.

Usage:
    python -m train.best_models config/kws/arch-mamba1-kws-2.yaml \
        config/har/arch-mamba1-har.yaml --n 10 --bits 8 --strat ptq
    python -m train.best_models config/kws/arch-mamba1-kws-2.yaml \
        --n 5 --bits 16 --strat ptq --sort latency
"""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

# method -> results key suffix (mirrors QUANTIZATION_METHODS in top_models.py)
_METHOD_SUFFIX = {"ptq": "", "tqt": "_strat"}
# precision -> results key suffix
_PRECISION_SUFFIX = {8: "", 16: "_int16"}

_VAL_KEY = "quantized_accuracy{suffix}"
_TEST_KEY = "test_quantized_accuracy{suffix}"
_PARAM_KEY = "nr_parameters"


def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding study name."""
    cfg = OmegaConf.load(config_path)
    name = f"{cfg.MODEL}-{cfg.DATASET}"
    if cfg.get("EXPERIMENT_NAME"):
        name += f"-{cfg.EXPERIMENT_NAME}"
    return name


def derive_keys(bits: int, strat: str) -> tuple[str, str]:
    """Return (validation_key, test_key) for the given precision × method."""
    suffix = _METHOD_SUFFIX[strat] + _PRECISION_SUFFIX[bits]
    return _VAL_KEY.format(suffix=suffix), _TEST_KEY.format(suffix=suffix)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the n best models across studies ranked by "
            "validation accuracy / cost metric (--sort)."
        )
    )
    parser.add_argument("configs", nargs="+",
                        help="Paths to Hydra config YAML files (one or more)")
    parser.add_argument("--sort", type=str, default="param",
                        choices=["param", "latency"],
                        help="Ranking metric: 'param' = validation accuracy / "
                             "nr_parameters (default), 'latency' = validation "
                             "accuracy / mcu_latency_ms (only trials with MCU "
                             "latency data are considered)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of top models to report (default: 10)")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16],
                        help="Quantization precision (default: 8)")
    parser.add_argument("--strat", type=str, default="ptq",
                        choices=["ptq", "tqt"],
                        help="Quantization method (default: ptq)")
    parser.add_argument("--min-val-acc", type=float, default=0.0,
                        help="Minimum validation accuracy (in %) to include a trial "
                             "(default: 0.0, i.e. no filtering)")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repository root (default: parent of this file's parent)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.bits == 16 and args.strat == "tqt":
        print("ERROR: 16-bit TQT quantization is not produced by the pipeline; "
              "no results exist for this combination.", file=sys.stderr)
        return 1

    repo_root = Path(args.repo_root) if args.repo_root \
        else Path(__file__).resolve().parent.parent
    val_key, test_key = derive_keys(args.bits, args.strat)
    sort_key = "val_acc_per_ms" if args.sort == "latency" else "val_acc_per_param"

    rows = []  # dicts keyed by field name
    for config_path in args.configs:
        study_name = study_name_from_config(config_path)
        results_path = repo_root / "experiments" / study_name / "results.json"
        if not results_path.exists():
            print(f"WARNING: no results.json for study '{study_name}' "
                  f"({results_path}), skipping.", file=sys.stderr)
            continue
        with open(results_path, "r") as f:
            results = json.load(f)

        n_used = 0
        for entry in results:
            tn = entry.get("trial_number")
            val_acc = entry.get(val_key)
            test_acc = entry.get(test_key)
            nr_params = entry.get(_PARAM_KEY)
            latency_ms = entry.get("mcu_latency_ms")
            if tn is None or val_acc is None or test_acc is None or nr_params is None:
                continue
            if nr_params <= 0:
                continue
            if val_acc < args.min_val_acc:
                continue
            if args.sort == "latency" and (latency_ms is None or latency_ms <= 0):
                continue
            row = {
                "study": study_name,
                "trial_number": tn,
                "validation_accuracy": val_acc,
                "nr_parameters": nr_params,
                "test_accuracy": test_acc,
                "val_acc_per_param": val_acc / nr_params,
                "test_acc_per_param": test_acc / nr_params,
            }
            if latency_ms is not None and latency_ms > 0:
                row["mcu_latency_ms"] = latency_ms
                row["val_acc_per_ms"] = val_acc / latency_ms
                row["test_acc_per_ms"] = test_acc / latency_ms
            rows.append(row)
            n_used += 1
        if n_used:
            print(f"Study '{study_name}': {n_used} trials with {args.bits}-bit/"
                  f"{args.strat} results.")
        else:
            print(f"Study '{study_name}': no trials with {args.bits}-bit/"
                  f"{args.strat} results.")

    if not rows:
        print("ERROR: no matching trials found across the given configs.",
              file=sys.stderr)
        return 1

    # Selection (min-accuracy filter + ranking) uses validation results only;
    # test accuracy is reported in the table for reference and never influences
    # which models are chosen.
    rows.sort(key=lambda r: r[sort_key], reverse=True)
    top = rows[: args.n]

    if args.sort == "latency":
        header = (
            f"{'study':<22}{'trial':>6}{'val_acc':>9}{'lat_ms':>10}"
            f"{'test_acc':>10}{'val/ms':>12}{'test/ms':>12}"
        )
    else:
        header = (
            f"{'study':<22}{'trial':>6}{'val_acc':>9}{'nr_params':>11}"
            f"{'test_acc':>10}{'val/nr':>12}{'test/nr':>12}"
        )
    print("\n" + header)
    print("-" * len(header))
    for r in top:
        if args.sort == "latency":
            print(
                f"{r['study']:<22}{r['trial_number']:>6}"
                f"{r['validation_accuracy']:>9.2f}"
                f"{r['mcu_latency_ms']:>10.2f}"
                f"{r['test_accuracy']:>10.2f}"
                f"{r['val_acc_per_ms']:>12.5f}"
                f"{r['test_acc_per_ms']:>12.5f}"
            )
        else:
            print(
                f"{r['study']:<22}{r['trial_number']:>6}"
                f"{r['validation_accuracy']:>9.2f}"
                f"{r['nr_parameters']:>11}"
                f"{r['test_accuracy']:>10.2f}"
                f"{r['val_acc_per_param']:>12.5f}"
                f"{r['test_acc_per_param']:>12.5f}"
            )
    print("-" * len(header))
    filter_note = (
        f" (min val acc {args.min_val_acc:.2f}%)" if args.min_val_acc > 0.0 else ""
    )
    sort_note = ("validation accuracy / mcu_latency_ms" if args.sort == "latency"
                 else "validation accuracy / nr_parameters")
    print(f"Ranked by {sort_note} "
          f"({args.bits}-bit, {args.strat}), top {args.n} of {len(rows)} total"
          f"{filter_note}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
