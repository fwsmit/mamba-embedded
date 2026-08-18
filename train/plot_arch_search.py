"""
HPO Comparison: compare multiple Optuna studies from their Hydra config files.
Pareto front and supporting analysis plots for N multi-objective studies.

Usage:
  python plot_arch_search.py config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py config/a.yaml config/b.yaml config/c.yaml config/d.yaml
  python plot_arch_search.py --plot pareto config/arch-mamba1-kws.yaml config/arch-mamba1-kws-multi.yaml
  python plot_arch_search.py --plot accuracy config/arch-mamba1-kws.yaml
  python plot_arch_search.py --plot accuracy --bar config/arch-mamba1-kws.yaml
  python plot_arch_search.py --plot pareto --use-mcu config/arch-mamba1-har.yaml
  python plot_arch_search.py --plot latency config/arch-mamba1-har.yaml
  python plot_arch_search.py --plot mcu_pareto --size 8 --quantization tqt config/arch-mamba1-har.yaml
  python plot_arch_search.py --plot importance config/arch-mamba1-har.yaml
"""

import argparse
import optuna
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
from .plot_types.stacked_profiling import create_stacked_profiling_plot
from .plot_types.common import create_out_dirs, resolve_accuracy
from .plot_types.generic_scatter import create_generic_scatter_plot
from .plot_types.accuracy import create_accuracy_comparison_bar_plot, create_accuracy_comparison_plot
from .plot_types.accuracy_grid import create_accuracy_grid_plot
from .plot_types.quantization_loss import create_quantization_loss_plot
from .plot_types.pareto_front import create_mcu_pareto_plot, create_pareto_front_plot
from .plot_types.latency_correlation import create_latency_correlation_plot
from .plot_types.param_accuracy import create_param_accuracy_plot
from .plot_types.param_importance import create_param_importance_plot

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


def load_results_data(studies_data, repo_root):
    """Load results.json for each study into sd['results_data']. Returns True if any data was loaded."""
    any_loaded = False
    for sd in studies_data:
        results_path = repo_root / "experiments" / sd["study_name"] / "results.json"
        if results_path.exists():
            # print(f"  Loading results from {results_path} …")
            with open(results_path) as f:
                sd["results_data"] = json.load(f)
            # print(f"    → {len(sd['results_data'])} trials")
            any_loaded = True
        else:
            print(f"  No results.json found at {results_path}, no MCU data for {sd['name']}.")
            sd["results_data"] = []
    return any_loaded


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
        "--plot", "-p", choices=["pareto", "accuracy", "accuracy_grid", "mcu_pareto", "latency", "scatter", "profiling", "stacked", "quantization_loss", "param_accuracy", "importance"], required=True,
        help="Which plot to create: 'pareto' (Pareto front comparison), "
             "'accuracy' (float vs quantized accuracy per study), "
             "'accuracy_grid' (combined accuracy subplot grid sharing axes), "
             "'mcu_pareto' (PC Pareto front with MCU-tested highlights + MCU perf plot; "
             "--size/--quantization select the accuracy variant), "
             "'latency' (PC latency vs MCU latency scatter plot), "
             "'scatter' (generic scatter plot of two results.json fields), "
             "'profiling' (MCU operator profiling bar chart for a specific trial), "
             "'stacked' (MCU operator latency stacked by trial across all MCU-tested trials), "
             "'quantization_loss' (quantization loss comparison across studies), or "
             "'param_accuracy' (nr of parameters vs accuracy with N models selected from the Pareto front; "
             "for HAR studies, literature reference points are overlaid), or "
             "'importance' (hyperparameter importance per study, one panel per objective, "
             "matching optuna-dashboard's PedAnova computation)."
    )
    parser.add_argument(
        "--bar", action="store_true",
        help="For --plot accuracy: draw a grouped bar chart (one bar group per "
             "trial) instead of the float-vs-quantized scatter plot."
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
        "--trial", type=int, default=None,
        help="Trial number for the 'profiling' plot."
    )
    parser.add_argument("--absolute", action="store_true", default=False,
                        help="For --plot stacked: plot summed latency (ms) instead of normalised %%.")
    parser.add_argument(
        "--use-mcu", action="store_true",
        help="Use MCU accuracy and latency from results.json instead of PC objectives "
             "in the Pareto front plot (only valid with --plot pareto)."
    )
    parser.add_argument(
        "--ylim", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="Fix the y-axis range (used with --plot quantization_loss). Pass "
             "the same values to comparable plots (e.g. KWS and HAR) to keep "
             "their y-axes consistent."
    )
    parser.add_argument(
        "--use-param-size", action="store_true",
        help="Correlate MCU latency with parameter size instead of PC latency "
             "(only valid with --plot latency)."
    )
    parser.add_argument(
        "--x-field", type=str, default=None,
        help="Field name in results.json for the x-axis (required for --plot scatter)."
    )
    parser.add_argument(
        "--y-field", type=str, default=None,
        help="Field name in results.json for the y-axis (required for --plot scatter)."
    )
    parser.add_argument(
        "--x-label", type=str, default="",
        help="X-axis label for the scatter plot (optional)."
    )
    parser.add_argument(
        "--y-label", type=str, default="",
        help="Y-axis label for the scatter plot (optional)."
    )
    parser.add_argument(
        "--y-zero-line", action="store_true",
        help="For --plot scatter: draw a dashed reference line at y=0 "
             "(useful e.g. for overfitting checks where positive values mean "
             "test accuracy is below validation accuracy)."
    )
    parser.add_argument(
        "--n-models", type=int, default=10,
        help="Number of models to select from the Pareto front for --plot "
             "param_accuracy (default: 10)."
    )
    parser.add_argument(
        "--min-val-acc", type=float, default=None,
        help="For --plot param_accuracy: only consider models whose "
             "validation-set accuracy is strictly above this value (percent) "
             "for selection (default: no threshold)."
    )
    parser.add_argument(
        "--size", type=int, choices=[32, 16, 8], default=8,
        help="Bit width for --plot param_accuracy and --plot mcu_pareto: 32 "
             "(float, unquantized), 16 (int16) or 8 (int8). Default: 8."
    )
    parser.add_argument(
        "--quantization", type=str, choices=["no", "percent", "tqt"], default="percent",
        help="Quantization method for --plot param_accuracy and --plot "
             "mcu_pareto: 'no' (float), 'percent' (standard PTQ) or 'tqt' "
             "(KL-TQT). Default: percent."
    )
    args = parser.parse_args()

    if args.bar and args.plot != "accuracy":
        parser.error("--bar is only valid with --plot accuracy")

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
    print(f"Creating {args.plot} plot")

    # ── Gather study data ─────────────────────────────────────────────────────
    # Sort by display name so the legend order and colour assignment are
    # consistent regardless of the order/expansion of the config arguments
    # (e.g. config/har/* vs config/kws/*).
    meta_list = sorted(
        (load_study_meta(cp) for cp in args.configs), key=lambda m: m["display_name"]
    )
    studies_data = []
    for i, meta in enumerate(meta_list):
        name = meta["study_name"]
        display_name = meta["display_name"]
        color_base, color_par = COLORS[i % len(COLORS)]
        # print(f"Study {i+1}: {name} → colour {color_base}")
        # print(f"  → Plot label: {display_name}")

        # print("  Loading study …")
        study = load_study(name)
        df = trials_df(study)
        mask = pareto_mask(df)

        # Sort Pareto front by latency for clean step-line drawing
        par = df[mask].copy().sort_values("latency")

        print(f"  {name}: {len(df)} trials, {len(par)} Pareto-optimal")

        studies_data.append({
            "name": display_name,
            "study_name": name,
            "study": study,
            "df": df,
            "par": par,
            "color": color_base,
            "color_par": color_par,
            "idx": i,
        })

    # ── Create the requested plot ───────────────────────────────────────────
    plot_created = False


    if args.plot == "accuracy":
        load_results_data(studies_data, repo_root)
        for sd in studies_data:
            if sd.get("results_data"):
                if args.bar:
                    create_accuracy_comparison_bar_plot(sd["name"], sd["results_data"], title)
                else:
                    create_accuracy_comparison_plot(sd["name"], sd["results_data"], title)
                plot_created = True
            else:
                print(f"  No results.json found for {sd['name']}, skipping.")

    elif args.plot == "accuracy_grid":
        load_results_data(studies_data, repo_root)
        create_accuracy_grid_plot(studies_data, title)
        plot_created = True

    elif args.plot == "mcu_pareto":
        try:
            accuracy_field, _, accuracy_label = resolve_accuracy(
                args.size, args.quantization)
        except ValueError as e:
            parser.error(str(e))
        load_results_data(studies_data, repo_root)
        create_mcu_pareto_plot(studies_data, title,
                               accuracy_field=accuracy_field,
                               accuracy_label=accuracy_label)
        plot_created = True

    elif args.plot == "latency":
        load_results_data(studies_data, repo_root)
        create_latency_correlation_plot(studies_data, title, use_param_size=args.use_param_size)
        plot_created = True

    elif args.plot == "pareto":
        if args.use_mcu:
            print("  Using MCU data from results.json instead of PC objectives ...")
            load_results_data(studies_data, repo_root)
            for sd in studies_data:
                if not sd.get("results_data"):
                    print(f"  Warning: No results.json for {sd['name']}, keeping PC data.")
                    continue
                mcu_rows = []
                for rd in sd["results_data"]:
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

    elif args.plot == "scatter":
        if args.x_field is None or args.y_field is None:
            parser.error("--x-field and --y-field are required when using --plot scatter")
        load_results_data(studies_data, repo_root)
        create_generic_scatter_plot(studies_data, title,
                                    x_field=args.x_field, y_field=args.y_field,
                                    x_label=args.x_label, y_label=args.y_label,
                                    y_zero_line=args.y_zero_line)
        plot_created = True

    elif args.plot == "quantization_loss":
        load_results_data(studies_data, repo_root)
        create_quantization_loss_plot(studies_data, title, ylim=args.ylim)
        plot_created = True

    elif args.plot == "param_accuracy":
        try:
            accuracy_field, selection_field, accuracy_label = resolve_accuracy(
                args.size, args.quantization)
        except ValueError as e:
            parser.error(str(e))
        load_results_data(studies_data, repo_root)
        create_param_accuracy_plot(studies_data, title,
                                   n_models=args.n_models,
                                   accuracy_field=accuracy_field,
                                   selection_accuracy_field=selection_field,
                                   accuracy_label=accuracy_label,
                                   min_val_acc=args.min_val_acc)
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

    elif args.plot == "importance":
        create_param_importance_plot(studies_data)
        plot_created = True

    elif args.plot == "stacked":
        # ── MCU operator latency stacked by trial across all MCU-tested trials ──
        if len(args.configs) != 1:
            parser.error("stacked plot requires exactly one config file.")

        meta = load_study_meta(args.configs[0])
        study_name = meta["study_name"]
        display_name = meta["display_name"]
        print(f"  Stacked profiling across trials from study {study_name}")

        if create_stacked_profiling_plot(study_name, args.configs[0], display_name,
                                         absolute=args.absolute):
            plot_created = True

    if not plot_created:
        print("\nNo plots created.")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
