#!/usr/bin/env python3
"""
Run every quantized Mamba-Lite model (.espdl) in experiments/mambalite-micro/
on the ESP32-S3 via run-esp.sh and store the MCU program output (serial
monitor output: latency, latency profile, memory profile) in a per-model
<model>.<suffix>.output file.

Build / flash output is only stored when a run fails (to record that failure);
successful runs only contain the program output.

Usage:
    conda run -n torch-pascal python -m train.run_lite_mcu
    conda run -n torch-pascal python -m train.run_lite_mcu --espdl experiments/mambalite-micro/har-single.espdl
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments" / "mambalite-micro"
RUN_SCRIPT = REPO_ROOT / "run-esp.sh"

# Give each run-esp.sh invocation generous headroom: the monitor inside has a
# 3000 s timeout of its own.
SUBPROCESS_TIMEOUT_S = 3600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espdl", nargs="*", default=None,
                        help="Explicit .espdl paths (default: all with a dataset bin)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when an .output file already exists")
    args = parser.parse_args()

    if args.espdl:
        files = sorted(Path(p) for p in args.espdl)
    else:
        files = sorted(EXPERIMENTS.glob("*.espdl"))
    files = [f for f in files if f.suffix == ".espdl"]

    # only models that have a matching dataset binary can run on the MCU
    files = [f for f in files if (f.parent / ("dataset-" + f.stem + ".bin")).exists()]
    if not files:
        print("No .espdl models with a matching dataset-<stem>.bin found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for espdl in files:
        out_path = espdl.with_suffix(".output")
        if out_path.exists() and not args.force:
            print(f"[skip] {espdl.name} (output already exists)")
            results.append((espdl.name, "skipped"))
            continue

        print(f"\n[run ] {espdl.name} ...", flush=True)
        try:
            result = subprocess.run(
                [str(RUN_SCRIPT), str(espdl)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=SUBPROCESS_TIMEOUT_S,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"

        with open(out_path, "w") as fh:
            fh.write(f"Exit code: {code}\n")
            fh.write("stdout:\n")
            fh.write(result.stdout if "result" in dir() else "(subprocess timed out)\n")
            fh.write("stderr:\n")
            fh.write(result.stderr if "result" in dir() else "(subprocess timed out)\n")
        print(f"[done] {espdl.name} -> exit {code}; output saved to {out_path}", flush=True)
        results.append((espdl.name, code))

    print("\n=== SUMMARY ===")
    for name, code in results:
        print(f"  {name:40s} {code}")

    n_ok = sum(1 for _, c in results if c == 0)
    print(f"\n{len(results)} runs, {n_ok} INFERENCE_OK.")


if __name__ == "__main__":
    main()