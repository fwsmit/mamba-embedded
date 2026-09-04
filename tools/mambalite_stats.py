#!/usr/bin/env python3
"""Report latency and the full memory profile from the Mamba-Lite
MCU run outputs.

Parses every ``experiments/mambalite-micro/*.output`` log produced when a
Mamba model is flashed to the ESP32-S3 (see ``run-mambalite-mcu.sh``) and
prints, per model:

  * average single-inference latency from the line
    ``Average single-inference latency: <us> us (<ms> ms)``
  * the full per-context memory profile (not only the totals): the
    ``fbs_model`` / ``-- parameter`` / ``parameter_copy`` / ``variable`` /
    ``others`` / ``total`` rows, each broken down by internal RAM / PSRAM /
    FLASH, plus the summed peak RAM and total flash.

This mirrors the parsing in ``train/plot_types/mambalite_lite.py`` but prints
tables instead of drawing figures.

Usage:
    python tools/mambalite_stats.py [EXPERIMENTS_DIR] [--filter KEYWORD]

Positional arguments:
    EXPERIMENTS_DIR   directory of run outputs (default:
                      experiments/mambalite-micro under this repo)
    --filter KEYWORD  only report runs whose filename contains KEYWORD
                      (e.g. ``har``, ``kws``, ``_int16``)
"""

import argparse
import os
import re
import sys

LAT_RE = re.compile(r"Average single-inference latency:\s*([\d.]+)\s*us")

# A single memory-summary cell value: "13.50KB" or "-- 13.50KB" or "".
CELL_RE = re.compile(r"[-\s]*(\d+\.\d+)\s*KB")


def _parse_cell(cell):
    """Return the KB value of a memory cell, or 0.0 when empty."""
    m = CELL_RE.search(cell)
    return float(m.group(1)) if m else 0.0


def _parse_output(text):
    """Extract latency and the full memory profile.

    Returns a dict with keys: latency_us, mem (list of (name, ir, psram,
    flash) rows).
    """
    lat_m = LAT_RE.search(text)
    if not lat_m:
        return None

    mem = []
    in_mem = False
    for line in text.splitlines():
        if "memory summary" in line:
            in_mem = True
            continue
        if not in_mem:
            continue
        # Memory summary rows: split on '|' -> [pre, name, internal, psram, flash].
        if line.count("|") >= 5 and "+---" not in line:
            cells = [c.strip() for c in line.split("|")]
            name = cells[1]
            # Skip the column-header row (name == "internal RAM").
            if name != "internal RAM":
                values = [_parse_cell(c) for c in cells[2:5]]
                mem.append((name, *values))
                if name == "total":
                    in_mem = False
    if not mem:
        return None
    return {"latency_us": float(lat_m.group(1)), "mem": mem}


def _fmt(v):
    return f"{v:.2f}"


def _print_full_memory(run):
    """One table per model: per-context internal/PSRAM/FLASH + totals."""
    rows = run["mem"]
    name_w = max(len(r[0]) for r in rows) + 1
    hdr = f"  {'context':<{name_w}}{'internal RAM':>14}{'PSRAM':>14}{'FLASH':>14}"
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, ir, ps, fl in rows:
        if name == "total":
            print("  " + "-" * (len(hdr) - 2))
        print(f"  {name:<{name_w}}{_fmt(ir):>14}{_fmt(ps):>14}{_fmt(fl):>14}")
    print("  " + "-" * (len(hdr) - 2))


def _summarize(run):
    tot = {r[0]: r for r in run["mem"]}["total"]
    ir, ps, fl = tot[1], tot[2], tot[3]
    return ir, ps, ir + ps, fl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("exp_dir", nargs="?", default=None,
                    help="directory of run outputs (default: "
                         "experiments/mambalite-micro)")
    ap.add_argument("--filter", default=None,
                    help="only report runs whose filename contains KEYWORD")
    args = ap.parse_args(argv)

    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exp_dir = os.path.join(repo_root, "experiments", "mambalite-micro")

    if not os.path.isdir(exp_dir):
        print(f"ERROR: no such directory: {exp_dir}")
        return 1

    files = sorted(f for f in os.listdir(exp_dir) if f.endswith(".output"))
    if args.filter:
        files = [f for f in files if args.filter in f]

    runs = []
    for fname in files:
        text = open(os.path.join(exp_dir, fname),
                    encoding="utf-8", errors="replace").read()
        run = _parse_output(text)
        if run is None:
            print(f"skipped (no result): {fname}")
            continue
        run["file"] = fname
        runs.append(run)

    if not runs:
        print("No run outputs parsed.")
        return 0

    # ---- Summary table ----
    print(f"\nParsed {len(runs)} run(s) from {exp_dir}\n")
    print("  ================= SUMMARY (latency + peak memory + flash) ================")
    hdr = (f"  {'file':<28}{'latency (us)':>14}{'latency (ms)':>14}"
           f"{'intRAM (KB)':>12}{'PSRAM (KB)':>12}{'peak RAM (KB)':>14}"
           f"{'flash (KB)':>12}")
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in runs:
        ir, ps, peak, fl = _summarize(r)
        print(f"  {r['file']:<28}{r['latency_us']:>14.1f}{r['latency_us'] / 1e3:>14.3f}"
              f"{ir:>12.2f}{ps:>12.2f}{peak:>14.2f}{fl:>12.2f}")
    print("  " + "-" * (len(hdr) - 2))

    # ---- Full memory profile per model ----
    for r in runs:
        print(f"\n  === {r['file']} ===")
        print(f"  Average single-inference latency: {r['latency_us']:.1f} us "
              f"({r['latency_us'] / 1e3:.3f} ms)")
        print("\n  -- Full memory profile (per context, KB) --")
        _print_full_memory(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())