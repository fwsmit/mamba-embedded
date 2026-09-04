#!/usr/bin/env python
"""Run int16-subset experiments through the espdl-quantize skill harness.

Each candidate is a set of op names promoted to int16. Runs them sequentially,
records fast-eval val_acc, and normalises names against the baseline.
"""
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/friso/Agents/mamba-embedded")
SKILL = REPO / ".pi" / "skills" / "espdl-quantize"
PY = "/home/friso/.conda/envs/torch-pascal/bin/python"
USER = SKILL / "contracts" / "kws_trial25" / "user_quant.py"
BMARK = REPO / "out" / "kws_trial25"

W = {
    "/linear_in/MatMul", "/mamba_layers.0/MatMul", "/mamba_layers.0/conv1d/Conv",
    "/mamba_layers.0/x_proj/MatMul", "/mamba_layers.0/MatMul_1",
    "/mamba_layers.0/out_proj/MatMul", "/classifier/Gemm",
}
ARITH = {"Mul", "Add", "ReduceSum", "Sub", "Sigmoid", "Relu", "Neg", "Exp",
         "Log", "Sqrt", "Div", "Pow", "MatMul", "Gemm", "Conv"}


def load_ops():
    d = json.load(open(BMARK / "iter_1" / "simplified_ops.json"))
    return d["ops"]


def promote(op, include_types, x_type=16):
    if op["name"] in W or not op["name"]:
        return False
    return op["op_type"] in include_types


def cand_full():
    return [o["name"] for o in load_ops() if o["name"] and o["name"] not in W]


def cand_arith():
    return [o["name"] for o in load_ops() if promote(o, ARITH)]


def cand_arith_gather():
    return [o["name"] for o in load_ops() if promote(o, ARITH | {"Gather"})]


def cand_core():
    return [o["name"] for o in load_ops() if promote(o, {"Mul", "Add", "ReduceSum"})]


def cand_no_shape():
    shape_types = {"Unsqueeze", "Reshape", "Slice"}
    return [o["name"] for o in load_ops() if o["name"] not in W and o["name"]
            and o["op_type"] not in shape_types]


def write_setting(outdir, names, calib):
    outdir.mkdir(parents=True, exist_ok=True)
    disp = [{"op": n, "bits": 16} for n in sorted(names)]
    setting = {"iteration_id": 1,
               "rationale": f"sweep subset ({len(names)} int16 ops)",
               "calib_algorithm": calib, "dispatching_table": disp}
    sp = outdir / "setting.json"
    json.dump(setting, open(sp, "w"), indent=2)
    return sp


def run_one(outdir, user_quant=None):
    r = subprocess.run(
        [PY, str(SKILL / "scripts" / "run_iteration.py"),
         "--user-quant", str(USER),
         "--setting", str(outdir / "setting.json"),
         "--output-dir", str(outdir)],
        capture_output=True, text=True, cwd=REPO)
    return r.stdout + r.stderr


def extract_acc(log):
    m = re.search(r"done in [0-9.]+s\. val_acc=np\.float64\(([0-9.]+)\)", log)
    if not m:
        m = re.search(r"done in [0-9.]+s\. val_acc=([0-9.]+)", log)
    return float(m.group(1)) if m else float("nan")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "sweep1"
    calib = sys.argv[2] if len(sys.argv) > 2 else "percentile"
    cands = {
        # baseline: scan-block only, 485 ops (the iter_1 winner)
        "scan_485": json.load(open("/tmp/scan_int16.json")),
        # arithmetic / recurrent only — no shape/index
        "arith": cand_arith(),
        "arith_gather": cand_arith_gather(),
        # reduce only the recurrent core (Mul/Add/ReduceSum)
        "core_state": cand_core(),
        "no_shape": cand_no_shape(),
    }
    metric = {}
    for name, names in cands.items():
        outdir = REPO / "out" / f"kws_{tag}" / name
        shutil.rmtree(outdir, ignore_errors=True)
        sp = write_setting(outdir, names, calib)
        print(f"[sweep] {name}: {len(names)} int16 ops", flush=True)
        log = run_one(outdir)
        (BMARK / f"sweep_{name}.log").write_text(log)
        acc = extract_acc(log)
        metric[name] = (len(names), acc)
        print(f"  -> {name}: val_acc={acc:.2f}  ({len(names)} int16)", flush=True)
    json.dump(metric, open(BMARK / f"sweep_{tag}.json", "w"), indent=2)
    print("DONE", json.dumps(metric, indent=2))


if __name__ == "__main__":
    main()