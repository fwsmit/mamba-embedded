#!/usr/bin/env python3
"""Wilcoxon signed-rank tests comparing the three quantization methods
(int8 PTQ, int16 PTQ, int8 KL-TQT) pairwise.

For every trial with all three methods evaluated, the per-trial metric is
computed for each method and the paired differences between two methods are
tested for a symmetric distribution around zero (the null hypothesis of the
Wilcoxon signed-rank test).

Two metrics are available:

  * ``loss`` (default): quantization loss = ``float_accuracy -
    quantized_accuracy`` (per trial, so differences in float model quality
    between trials do not confound the comparison). Lower is better.
  * ``raw``: the quantized accuracy itself. Higher is better.

By default the test runs on the validation set, the test set, and both, and
for three groups: all trials pooled, HAR-only and KWS-only (studies are
assigned to a dataset by their ``mamba-1-har*`` / ``mamba-1-kws*`` names).

Trials where the KL-TQT run diverged (``*_strat`` accuracy recorded as 0, no
``.espdl`` produced) are excluded by default because a sentinel 0 is not a
real measurement; pass ``--keep-diverged`` to include them anyway.

Two modes are available:

  * ``--mode pairs`` (default): pairwise Wilcoxon signed-rank tests between
    the three methods, as described below.
  * ``--mode float``: each method is compared against the floating point
    model instead. The per-trial delta is the quantization loss
    (``float_accuracy - quantized_accuracy``); the median delta, its
    Hodges-Lehmann estimate, the IQR of the deltas and a one-sample Wilcoxon
    test of "delta = 0" are reported per method.

Usage:
    python tools/wilcoxon_test.py [EXPERIMENTS_DIR ...] [--group all|har|kws]
        [--metric loss|raw] [--set val|test|both] [--mode pairs|float]
        [--keep-diverged]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

METHODS = [("int8 PTQ", "quantized_accuracy"),
           ("int16 PTQ", "quantized_accuracy_int16"),
           ("int8 KL-TQT", "quantized_accuracy_strat")]
PAIRS = [(0, 1), (0, 2), (1, 2)]
ALPHA = 0.05


def load_trials(dirs, keep_diverged):
    trials = []
    excluded = []
    for d in dirs:
        data = json.load(open(os.path.join(d, "results.json")))
        study = os.path.basename(d.rstrip(os.sep))
        dataset = "har" if study.startswith("mamba-1-har") else "kws"
        for entry in data:
            if any(entry.get(key) is None or entry.get("test_" + key) is None
                   for _, key in METHODS):
                continue
            if (entry.get("quantized_accuracy_strat") == 0
                    or entry.get("test_quantized_accuracy_strat") == 0):
                if keep_diverged:
                    excluded.append((study, entry["trial_number"],
                                     "diverged KL-TQT kept (strat 0)"))
                else:
                    excluded.append((study, entry["trial_number"],
                                     "diverged KL-TQT excluded"))
                    continue
            trials.append({
                "study": study, "dataset": dataset, "trial": entry["trial_number"],
                "float": entry["float_accuracy"],
                "test_float": entry["test_float_accuracy"],
                "val": {k: entry[k] for _, k in METHODS},
                "test": {k: entry["test_" + k] for _, k in METHODS},
            })
    return trials, excluded


def metric_array(trials, set_name, method_idx, metric):
    name, key = METHODS[method_idx]
    if set_name == "validation":
        vals = [t["val"][key] for t in trials]
        floats = [t["float"] for t in trials]
    else:
        vals = [t["test"][key] for t in trials]
        floats = [t["test_float"] for t in trials]
    if metric == "loss":
        return np.array([f - v for f, v in zip(floats, vals)])
    return np.array(vals)


def hodges_lehmann(x):
    """Hodges-Lehmann estimator: median of the pairwise averages (i <= j).
    The location estimator associated with the Wilcoxon signed-rank test,
    applied here to the paired differences A - B as a robust effect size."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.nan
    vals = np.empty(n * (n + 1) // 2)
    k = 0
    for i in range(n):
        vals[k:k + n - i] = (x[i] + x[i:]) / 2.0
        k += n - i
    return np.median(vals)


def run_test(values_a, values_b):
    stats = wilcoxon(values_a, values_b, zero_method="wilcox",
                     alternative="two-sided", method="exact")
    return stats.statistic, stats.pvalue


def report(trials, set_name, metric, groups):
    better = "lower" if metric == "loss" else "higher"
    print(f"=== {set_name.capitalize()} set — metric: {metric} "
          f"({'float - quantized accuracy, ' + better + ' is better' if metric == 'loss' else 'quantized accuracy, ' + better + ' is better'}) ===")
    for group in groups:
        sel = trials if group == "all" else [t for t in trials if t["dataset"] == group]
        if not sel:
            continue
        print(f"\n[{group}] n = {len(sel)} trials")
        header = (f"{'pair':<18} {'med A':>8} {'med B':>8} {'med Δ':>8} "
                  f"{'HL Δ':>8} {'IQR Δ':>8} {'W':>7} {'p':>10}  verdict")
        print(header)
        print("-" * len(header))
        for i, j in PAIRS:
            name_a, _ = METHODS[i]
            name_b, _ = METHODS[j]
            va = metric_array(sel, set_name, i, metric)
            vb = metric_array(sel, set_name, j, metric)
            diffs = va - vb
            W, p = run_test(va, vb)
            hl = hodges_lehmann(diffs)
            iqr = float(np.percentile(diffs, 75) - np.percentile(diffs, 25))
            gains = diffs < 0 if metric == "loss" else diffs > 0
            verdict = name_a if gains.sum() >= len(sel) / 2 else name_b
            if p < ALPHA:
                verdict += f" wins (p<{ALPHA})"
            print(f"{name_a + ' vs ' + name_b:<18} {np.median(va):>8.3f} "
                  f"{np.median(vb):>8.3f} {np.median(diffs):>8.3f} {hl:>8.3f} "
                  f"{iqr:>8.3f} {W:>7.0f} {p:>10.4g}  {verdict}")
        counts = [0] * len(METHODS)
        for t in sel:
            scores = [metric_array([t], set_name, k, metric)[0] for k in range(len(METHODS))]
            counts[int(np.argmin(scores) if metric == "loss" else np.argmax(scores))] += 1
        print("  trials where each method is best: "
              + ", ".join(f"{name}: {c}" for (name, _), c in zip(METHODS, counts)))
    print()


def report_vs_float(trials, set_name, metric, groups):
    print(f"=== {set_name.capitalize()} set — each method vs float "
          f"({'loss = float - quantized accuracy (positive = drop)' if metric == 'loss' else 'quantized accuracy'}) ===")
    for group in groups:
        sel = trials if group == "all" else [t for t in trials if t["dataset"] == group]
        if not sel:
            continue
        print(f"\n[{group}] n = {len(sel)} trials")
        header = (f"{'method':<14} {'med Δ':>8} {'HL Δ':>8} {'IQR Δ':>8} "
                  f"{'W':>7} {'p':>10}  verdict")
        print(header)
        print("-" * len(header))
        for k, (name, _) in enumerate(METHODS):
            deltas = metric_array(sel, set_name, k, metric)
            stats = wilcoxon(deltas, zero_method="wilcox",
                             alternative="two-sided", method="exact")
            W, p = stats.statistic, stats.pvalue
            hl = hodges_lehmann(deltas)
            iqr = float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
            med = float(np.median(deltas))
            if metric == "loss":
                verdict = f"drop > 0 (p<{ALPHA})" if p < ALPHA else "drop not signif."
            else:
                verdict = f"Δ ≠ 0 (p<{ALPHA})" if p < ALPHA else "Δ not signif."
            print(f"{name:<14} {med:>8.3f} {hl:>8.3f} {iqr:>8.3f} {W:>7.0f} {p:>10.4g}  {verdict}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*", default=None,
                    help="experiment directories with results.json "
                         "(default: all experiments/ dirs containing results.json)")
    ap.add_argument("--group", nargs="+", default=["all", "har", "kws"],
                    choices=["all", "har", "kws"])
    ap.add_argument("--metric", default="loss", choices=["loss", "raw"])
    ap.add_argument("--set", default="both", choices=["val", "test", "both"])
    ap.add_argument("--mode", default="pairs", choices=["pairs", "float"],
                    help="pairs = methods vs each other (default); "
                         "float = each method vs the float model")
    ap.add_argument("--keep-diverged", action="store_true",
                    help="keep trials where KL-TQT diverged (strat accuracy 0)")
    args = ap.parse_args()

    dirs = args.dirs or sorted(os.path.dirname(p)
                               for p in glob.glob("experiments/*/results.json"))
    if not dirs:
        sys.exit("no experiments found")

    trials, excluded = load_trials(dirs, args.keep_diverged)
    dropped = [e for e in excluded if "excluded" in e[2]]
    if dropped:
        print(f"Excluded {len(dropped)} trial(s) with diverged KL-TQT (strat=0):",
              file=sys.stderr)
        for study, tn, why in dropped:
            print(f"  {study}/trial {tn}", file=sys.stderr)
        print(file=sys.stderr)

    groups = list(dict.fromkeys(args.group))
    if args.set in ("val", "both"):
        (report_vs_float if args.mode == "float" else report)(
            trials, "validation", args.metric, groups)
    if args.set in ("test", "both"):
        (report_vs_float if args.mode == "float" else report)(
            trials, "test", args.metric, groups)


if __name__ == "__main__":
    main()