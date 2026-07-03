#!/usr/bin/env python3
"""Compare activation distributions between good and bad quantisation trials."""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.inspect_activations import add_intermediate_outputs, analyse_tensor, load_val_data, infer_dataset


def get_top_stats(model_path: Path, n_samples: int = 32):
    dataset = infer_dataset(model_path)
    model = onnx.load(str(model_path))
    aug = add_intermediate_outputs(model)
    tmp = model_path.with_suffix(".tmp-aug.onnx")
    onnx.save(aug, str(tmp))

    xs, _ = load_val_data(dataset, n_samples)
    sess = ort.InferenceSession(str(tmp))
    out_names = [o.name for o in aug.graph.output]

    all_int = defaultdict(list)
    for i in range(min(len(xs), 10)):
        outputs = sess.run(out_names, {"input": xs[i : i + 1]})
        for name, arr in zip(out_names, outputs):
            if name != "output":
                all_int[name].append(arr)

    tmp.unlink()

    stats = []
    for name, arrays in all_int.items():
        try:
            cat = np.concatenate(arrays, axis=0)
        except Exception:
            continue
        if cat.size == 0:
            continue
        info = analyse_tensor(name, cat)
        stats.append(info)
    return stats


trials = [
    (1, "GOOD: 0.47% drop"),
    (18, "BAD: 76.54% drop"),
    (0, "BAD: 82.52% drop"),
]

for trial, label in trials:
    p = Path(
        f"/home/friso/Agents/mamba-embedded/experiments/mamba-1-har-bidir-mul/mamba-1-har-bidir-mul-trial-{trial}.onnx"
    )
    if not p.exists():
        print(f"{p} not found")
        continue

    stats = get_top_stats(p)

    # --- ch_range_ratio top 10 ---
    stats.sort(key=lambda s: s["ch_range_ratio"], reverse=True)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Total tensors: {len(stats)}")
    print(f"\n  Top 10 by ch_range_ratio:")
    print(f"  {'Tensor name':<55s} {'Shape':<16s} {'Ratio':>10s}  {'Range':>20s}")
    print(f"  {'-'*55} {'-'*16} {'-'*10}  {'-'*20}")
    for s in stats[:10]:
        shape_str = "×".join(str(d) for d in s["shape"])
        print(
            f"  {s['name'][:54]:<55s} {shape_str:<16s} {s['ch_range_ratio']:>10.4g}  [{s['min']:<8.4g},{s['max']:<8.4g}]"
        )

    # --- ch_absmax_ratio top 10 ---
    stats_abs = sorted(stats, key=lambda s: s["ch_absmax_ratio"], reverse=True)
    print(f"\n  Top 10 by ch_absmax_ratio:")
    print(f"  {'Tensor name':<55s} {'Shape':<16s} {'Ratio':>10s}  {'Absmax range':>20s}")
    print(f"  {'-'*55} {'-'*16} {'-'*10}  {'-'*20}")
    for s in stats_abs[:10]:
        shape_str = "×".join(str(d) for d in s["shape"])
        print(
            f"  {s['name'][:54]:<55s} {shape_str:<16s} {s['ch_absmax_ratio']:>10.4g}  [{s['ch_absmax_min']:<8.4g},{s['ch_absmax_max']:<8.4g}]"
        )

    # --- Summary stats ---
    ch_ratios = [s["ch_absmax_ratio"] for s in stats]
    flagged = sum(1 for r in ch_ratios if r > 5.0)
    severe = sum(1 for r in ch_ratios if r > 50.0)
    extreme = sum(1 for r in ch_ratios if r > 500.0)
    print(f"\n  ch_absmax_ratio distribution:")
    print(f"    > 5.0  (flagged):  {flagged:>4d}")
    print(f"    > 50   (severe):   {severe:>4d}")
    print(f"    > 500  (extreme):  {extreme:>4d}")
    print(f"    max:               {max(ch_ratios):>10.4g}")
    print(f"    mean:              {np.mean(ch_ratios):>10.4g}")
    print(f"    median:            {np.median(ch_ratios):>10.4g}")
