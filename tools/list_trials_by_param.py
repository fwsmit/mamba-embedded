#!/usr/bin/env python3
"""List Optuna trials whose hidden dimension (d_model) equals a given value.

By default queries mamba_hpo.db for trials with d_model = 64. Prints each match
with its study, expand factor, and all other recorded hyperparameters.

Usage:
    python tools/list_trials_by_param.py [--db PATH] [--dim VALUE] [--with-values]

The search space configs in config/ use ``d_model`` as the hidden dimension,
so that is the parameter matched unless a search space names it ``hidden_dim``.
"""

import argparse
import sqlite3
from collections import defaultdict

# Parameters of interest, shown first in a fixed column order.
PREFERRED_ORDER = ["d_model", "expand", "d_state", "d_conv", "n_layers",
                   "nheads", "optimizer", "lr"]


def list_trials(db_path: str, dim: float, with_values: bool) -> None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    params = defaultdict(dict)
    for trial_id, name, value in cur.execute(
        "SELECT trial_id, param_name, param_value FROM trial_params"
    ):
        params[trial_id][name] = value

    trial_info = {
        tid: (study_id, number, state)
        for tid, study_id, number, state in cur.execute(
            "SELECT trial_id, study_id, number, state FROM trials"
        )
    }
    study_info = dict(cur.execute(
        "SELECT study_id, study_name FROM studies"
    ))
    values = defaultdict(list)
    if with_values:
        for trial_id, obj, value in cur.execute(
            "SELECT trial_id, objective, value FROM trial_values"
        ):
            values[trial_id].append((obj, value))

    # Collect matches: hidden dim under either name, equal to `dim`.
    matches = []
    for trial_id, p in params.items():
        hid = p.get("d_model", p.get("hidden_dim"))
        if hid is not None and abs(float(hid) - dim) < 1e-9:
            matches.append(trial_id)

    if not matches:
        print(f"No trials found with hidden dimension = {dim:g} in {db_path}.")
        conn.close()
        return

    # Column order: preferred params first, then any others in each trial.
    all_names = []
    for tid in matches:
        for name in params[tid]:
            if name not in all_names:
                all_names.append(name)
    order = [n for n in PREFERRED_ORDER if n in all_names] + \
            [n for n in all_names if n not in PREFERRED_ORDER]

    print(f"Trials with hidden dimension = {dim:g} ({db_path})\n")
    for tid in sorted(matches):
        study_id, number, state = trial_info[tid]
        study = study_info.get(study_id, "?")
        base = f"  study={study!r:<30} trial=#{number}  state={state}"
        for name in order:
            if name in params[tid]:
                base += f"  {name}={params[tid][name]:g}"
        print(base)
        if values[tid]:
            for obj, val in sorted(values[tid]):
                print(f"      objective[{obj}] = {val:g}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mamba_hpo.db",
                        help="Path to the Optuna SQLite DB (default mamba_hpo.db)")
    parser.add_argument("--dim", type=float, default=64.0,
                        help="Hidden dimension to match (default 64)")
    parser.add_argument("--with-values", action="store_true",
                        help="Also print trial objective values (latency, accuracy)")
    args = parser.parse_args()
    list_trials(args.db, args.dim, args.with_values)