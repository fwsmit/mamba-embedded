"""
HPO Comparison: compare multiple Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for N multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py config/a.yaml config/b.yaml config/c.yaml config/d.yaml
  python plot_arch_search.py --plot pareto config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py --plot accuracy config/arch-mamba1-kws.yaml
  python plot_arch_search.py --plot pareto --use-mcu config/arch-mamba1-har.yaml
  python plot_arch_search.py --plot latency config/arch-mamba1-har.yaml
"""

import argparse
import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
from omegaconf import OmegaConf
import json
import warnings

from .plot_types.profiling import create_profiling_plot
from .plot_types.common import create_out_dirs
from .plot_types.param_accuracy import create_param_vs_accuracy_plot
from .plot_types.param_latency import create_param_vs_latency_plot
from .plot_types.param_latency import create_param_vs_latency_plot
from .plot_types.accuracy import create_accuracy_comparison_plot
from .plot_types.quantization_loss import create_quantization_loss_plot
from .plot_types.pareto_front import create_mcu_pareto_plot, create_pareto_front_plot

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Output directory ──────────────────────────────────────────────────────────

# ── Config (defaults, overridden by command-line args) ───────────────────────
DB_URL       = "sqlite:///mamba_hpo.db"

# Colour palette for up to N studies (base, pareto)
COLORS = [
    ("#4C9BE8", "#1A5FA8"),   # blue
    ("#E8834C", "#A84F1A"),   # orange
    ("#4CAF50", "#2E7D32"),   # green
    ("#9C27B0", "#6A1B9A"),   # purple
    ("#009688", "#00695C"),   # teal
    ("#E91E63", "#AD1457"),   # pink
]

# ── Load studies ──────────────────────────────────────────────────────────────
def load_study(name):
    return optuna.load_study(study_name=name, storage=DB_URL)

def trials_df(study):
    """Return a DataFrame of completed trials with objectives and params."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(t.params)
        row["accuracy"] = t.values[0]   # objective 0
        row["latency"]  = t.values[1]   # objective 1
        row["number"]   = t.number
        rows.append(row)
    return pd.DataFrame(rows)

def pareto_mask(df):
    """Boolean mask for Pareto-optimal trials (max accuracy, min latency)."""
    acc = df["accuracy"].values
    lat = df["latency"].values
    n   = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good on both and strictly better on one
            if acc[j] >= acc[i] and lat[j] <= lat[i] and (acc[j] > acc[i] or lat[j] < lat[i]):
                dominated[i] = True
                break
    return ~dominated




def study_name_from_config(config_path: str) -> str:
    """Load a Hydra config YAML and return the corresponding Optuna study name."""
    cfg = OmegaConf.load(config_path)
    return f"{cfg.MODEL}-{cfg.DATASET}-{cfg.EXPERIMENT_NAME}" if cfg.get("EXPERIMENT_NAME") else f"{cfg.MODEL}-{cfg.DATASET}"


def load_study_meta(config_path: str) -> dict:
    """Load a Hydra config YAML and return metadata: study_name, display_name.

    The display name is set from the optional ``plot_description`` field in the
    config file (useful for giving configs clearer labels in Pareto front plots
    when comparing multiple experiments). Falls back to the auto-generated study
    name if ``plot_description`` is not present.
    """
    cfg = OmegaConf.load(config_path)
    study_name = study_name_from_config(config_path)
    display_name = cfg.get("plot_description") or study_name
    return {"study_name": study_name, "display_name": display_name}


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Optuna studies from their Hydra config files"
    )
    parser.add_argument(
        "configs", nargs="+",
        help="Paths to Hydra config YAML files (at least 1, up to any number)"
    )
    parser.add_argument(
        "--plot", "-p", choices=["param_vs_accuracy", "pareto", "accuracy", "mcu_pareto", "latency", "param_vs_latency", "profiling", "quantization_loss"], required=True,
        help="Which plot to create: 'pareto' (Pareto front comparison), "
             "'accuracy' (float vs quantized accuracy per study), "
             "'mcu_pareto' (PC Pareto front with MCU-tested highlights + MCU perf plot), "
             "'latency' (PC latency vs MCU latency scatter plot), "
             "'param_vs_latency' (parameter size vs MCU latency scatter plot), "
             "'profiling' (MCU operator profiling bar chart for a specific trial), or "
             "'quantization_loss' (quantization loss comparison across studies)."
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Title of the plot and base name for saved files. "
             "If not provided, derived from the study names."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the plot(s) on screen (default: only save to disk)."
    )
    parser.add_argument(
        "--mcu", action="store_true",
        help="Include on-device MCU accuracy bars in the accuracy plot (default: off)."
    )
    parser.add_argument(
        "--trial", type=int, default=None,
        help="Trial number for the 'profiling' plot."
    )
    parser.add_argument(
        "--use-mcu", action="store_true",
        help="Use MCU accuracy and latency from results.json instead of PC objectives "
             "in the Pareto front plot (only valid with --plot pareto)."
    )
    args = parser.parse_args()

    n = len(args.configs)
    if n == 0:
        parser.error("Provide at least one config path.")

    if n > len(COLORS):
        print(f"Warning: {n} studies provided but only {len(COLORS)} colours defined. "
              f"Colours will be recycled from the beginning.")

    repo_root = Path(__file__).resolve().parent.parent

    # ── Derive title if not provided ─────────────────────────────────────────
    title = args.title or " vs ".join(
        study_name_from_config(cp) for cp in args.configs
    )

    create_out_dirs()

    # ── Gather study data ─────────────────────────────────────────────────────
    studies_data = []
    for i, config_path in enumerate(args.configs):
        meta = load_study_meta(config_path)
        name = meta["study_name"]
        display_name = meta["display_name"]
        color_base, color_par = COLORS[i % len(COLORS)]
        print(f"Study {i+1}: {name}  (from {config_path})  → colour {color_base}")
        if display_name != name:
            print(f"  → Plot label: {display_name}")

        print("  Loading study …")
        study = load_study(name)
        df = trials_df(study)
        mask = pareto_mask(df)

        # Sort Pareto front by latency for clean step-line drawing
        par = df[mask].copy().sort_values("latency")

        print(f"  {name}: {len(df)} trials, {len(par)} Pareto-optimal")

        studies_data.append({
            "name": display_name,
            "study_name": name,
            "df": df,
            "par": par,
            "color": color_base,
            "color_par": color_par,
            "idx": i,
        })

    # ── Create the requested plot ───────────────────────────────────────────
    plot_created = False

    if args.plot == "accuracy":
        # ── Accuracy comparison (inferred from config) ────────────────────────
        for sd in studies_data:
            name = sd["name"]
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading accuracy results from {results_path} …")
                with open(results_path) as f:
                    results_data = json.load(f)
                create_accuracy_comparison_plot(name, results_data, title, show_mcu=args.mcu)
                plot_created = True
            else:
                print(f"  No results.json found at {results_path}, skipping.")

    elif args.plot == "mcu_pareto":
        # ── MCU Pareto plot (PC front + MCU performance) ────────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_mcu_pareto_plot(studies_data, title)
        plot_created = True

    elif args.plot == "latency":
        # ── PC latency vs MCU latency scatter plot ─────────────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_latency_correlation_plot(studies_data, title)
        plot_created = True

    elif args.plot == "pareto":
        # ── Pareto front plot (optionally with MCU data) ────────────────────
        if args.use_mcu:
            print("  Using MCU data from results.json instead of PC objectives ...")
            for sd in studies_data:
                results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
                if not results_path.exists():
                    print(f"  Warning: No results.json at {results_path}, cannot use MCU data for {sd['name']}.")
                    continue
                with open(results_path) as f:
                    results_data = json.load(f)
                mcu_rows = []
                for rd in results_data:
                    mcu_acc = rd.get("mcu_accuracy", np.nan)
                    mcu_lat = rd.get("mcu_latency_ms", np.nan)
                    if not np.isnan(mcu_acc) and not np.isnan(mcu_lat):
                        mcu_rows.append({
                            "accuracy": mcu_acc / 100.0,  # % → fraction
                            "latency": mcu_lat,
                            "number": rd["trial_number"],
                        })
                if not mcu_rows:
                    print(f"  Warning: No valid MCU results for {sd['name']}, keeping PC data.")
                    continue
                mcu_df = pd.DataFrame(mcu_rows)
                mask = pareto_mask(mcu_df)
                par = mcu_df[mask].copy().sort_values("latency")
                print(f"  {sd['name']}: {len(mcu_df)} MCU-tested trials, {len(par)} Pareto-optimal")
                sd["df"] = mcu_df
                sd["par"] = par

        create_pareto_front_plot(studies_data, title, use_mcu=args.use_mcu)
        plot_created = True

    elif args.plot == "param_vs_latency":
        # ── Parameter size vs MCU latency scatter plot ──────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_param_vs_latency_plot(studies_data, title)
        plot_created = True

    elif args.plot == "param_vs_accuracy":
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading MCU results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} MCU-tested trials")
            else:
                print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
                sd["results_data"] = []

        create_param_vs_accuracy_plot(studies_data, title)
        plot_created = True


    elif args.plot == "quantization_loss":
        # ── Quantization loss comparison across studies ────────────────────
        for sd in studies_data:
            results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
            if results_path.exists():
                print(f"  Loading results from {results_path} …")
                with open(results_path) as f:
                    sd["results_data"] = json.load(f)
                print(f"    → {len(sd['results_data'])} trials")
            else:
                print(f"  No results.json found at {results_path}, skipping {sd['name']}.")
                sd["results_data"] = []

        create_quantization_loss_plot(studies_data, title)
        plot_created = True

    elif args.plot == "profiling":
        # ── MCU operator profiling bar chart for a specific trial ──────────
        if args.trial is None:
            parser.error("--trial is required when using --plot profiling")
        if len(args.configs) != 1:
            parser.error("profiling plot requires exactly one config file.")

        meta = load_study_meta(args.configs[0])
        study_name = meta["study_name"]
        display_name = meta["display_name"]
        print(f"  Profiling trial {args.trial} from study {study_name}")

        if create_profiling_plot(study_name, args.configs[0], args.trial, display_name):
            plot_created = True

    if not plot_created:
        print("\nNo plots created.")
    else:
        print("\nAll requested figures saved.")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
