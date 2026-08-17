"""
Parameter-efficiency scatter plot: number of parameters (x) vs test-set
accuracy (y), with the N best models selected from the Pareto front of
(min parameters, max validation-set accuracy) and highlighted.

Usage (via plot_arch_search.py):
  python -m train.plot_arch_search --plot param_accuracy \\
      --n-models 10 --size 8 --quantization percent config/har/*.yaml

When the studies target the HAR dataset, reference points from the literature
(har-numbers-literature.md, see HAR_LITERATURE_POINTS below) are overlaid on
the same axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

from .common import savefig

ALPHA_SELECTED = 0.95
MARKER_SELECTED = "D"


# ── Literature reference points (overlaid for HAR studies) ──────────────────
# Sources collected in har-numbers-literature.md. To add a new point, append a
# dict with these keys:
#   name      : short label used for the annotation and the legend
#   params    : number of parameters (scalar)
#   value     : reported accuracy in percent (or F1-score, see metric)
#   metric    : "accuracy" or "f1" — f1 points get a distinct marker and an
#               "(F1, …)" suffix in their annotation so they are not mistaken
#               for accuracy numbers
#   source    : free-text citation (arXiv id / paper title)
#   cost      : optional compute cost in millions (MACs or FLOPs, see
#               cost_unit); shown in the annotation as e.g. "9.3M MACs"
#   cost_unit : "MACs" or "FLOPs" (only meaningful with ``cost``)
#   note      : optional free-text nuance (e.g. "average across datasets");
#               shown in the annotation
LIT_COLOR = "#111111"
LIT_MARKERS = {"accuracy": "X", "f1": "P"}
HAR_LITERATURE_POINTS = [
    dict(name="Novac et al.", params=3958, value=92.41, metric="accuracy",
         source="arXiv:2105.13331"),
    dict(name="MicrobiconvLSTM", params=11400, value=93.41, metric="accuracy",
         source="arXiv:2602.06523", cost=0.42, cost_unit="MACs",
         note="average across datasets"),
    dict(name="Machar", params=67380, value=99.32, metric="accuracy",
         source="arXiv:2602.06523", cost=10.37, cost_unit="FLOPs"),
    dict(name="Crossover-BiDir-BabyMamba", params=27000, value=95.10,
         metric="f1", source="BabyMamba-HAR", cost=2.21, cost_unit="MACs"),
    dict(name="CI-BabyMamba-HAR", params=28000, value=85.80, metric="f1",
         source="BabyMamba-HAR", cost=50.92, cost_unit="MACs"),
    dict(name="TinyHAR", params=55000, value=96.53, metric="f1",
         source="BabyMamba-HAR", cost=9.29, cost_unit="MACs"),
    dict(name="TinierHAR", params=33000, value=96.37, metric="f1",
         source="BabyMamba-HAR", cost=1.73, cost_unit="MACs"),
    dict(name="DeepConvLSTM", params=136000, value=93.53, metric="f1",
         source="BabyMamba-HAR", cost=15.51, cost_unit="MACs"),
]


def pareto_mask(x, y):
    """Boolean mask of Pareto-optimal points for objectives (min x, max y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is no worse on both and strictly better on one
            if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return ~dominated


def select_n_front(x, y, n):
    """Indices of up to *n* models spread evenly along the Pareto front."""
    order = np.where(pareto_mask(x, y))[0]
    order = order[np.argsort(x[order])]  # ascending nr of parameters
    k = min(n, len(order))
    if k <= 1:
        return order[:k]
    positions = np.linspace(0, len(order) - 1, k).round().astype(int)
    return order[np.unique(positions)]


def create_param_accuracy_plot(studies_data, title, n_models=10,
                               accuracy_field="test_quantized_accuracy",
                               selection_accuracy_field=None,
                               accuracy_label=None,
                               min_val_acc=None):
    """
    Scatter plot of number of parameters vs accuracy for the N most
    parameter-efficient models, selected from the combined Pareto front
    across all studies (max validation accuracy, min parameters) and plotted
    with diamond markers and trial-number annotations at their test-set
    accuracy.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name', 'study_name', 'df', 'par', 'results_data'
        (list of entries from results.json), 'color', 'color_par', 'idx'.
    title : str
        Used in the plot title and saved file names.
    n_models : int
        Number of models to select from the combined Pareto front across
        all studies.
    accuracy_field : str
        results.json field used for the y-axis accuracy (e.g.
        ``test_quantized_accuracy``, ``test_quantized_accuracy_int16``,
        ``test_quantized_accuracy_strat``, ``test_float_accuracy``).
    selection_accuracy_field : str, optional
        results.json field used to select the N models from the Pareto front
        (validation-set counterpart of ``accuracy_field``).  Defaults to
        ``accuracy_field`` when not given.
    accuracy_label : str, optional
        Y-axis label.  Falls back to the field name when not given.
    min_val_acc : float, optional
        Only models whose validation-set accuracy is strictly above this value
        (percent) are eligible for selection; models below it are not plotted.
        Defaults to no threshold.

    When any study targets the HAR dataset, reference points from the
    literature (see HAR_LITERATURE_POINTS) are overlaid as black markers:
    accuracy values as crosses, F1-scores as pentagons annotated with an
    "(F1)" suffix.
    """
    sel_field = selection_accuracy_field or accuracy_field

    print("Selecting from", sel_field)
    fig, ax = plt.subplots(figsize=(9, 7))

    # Collect all data points across studies, remembering their origin.
    # Row format: (study index into studies_data, nr_parameters, test
    # accuracy, validation accuracy, trial number)
    all_rows = []
    for sd_idx, sd in enumerate(studies_data):
        if not sd.get("results_data"):
            continue
        for rd in sd["results_data"]:
            npar = rd.get("nr_parameters", np.nan)
            acc = rd.get(accuracy_field, np.nan)
            sel_acc = rd.get(sel_field, np.nan)
            if np.isnan(npar) or np.isnan(acc):
                continue
            all_rows.append((sd_idx, npar, acc, sel_acc, rd.get("trial_number", -1)))

    if not all_rows:
        print(f"  No data found for field '{accuracy_field}' across all studies.")
        return

    # Select the N models from the combined Pareto front across all studies
    # (max validation accuracy, min params), restricted to models scoring
    # above the validation-set threshold (if any)
    x_all = np.array([r[1] for r in all_rows])
    sel_y_all = np.array([r[3] for r in all_rows])
    valid_sel = ~np.isnan(sel_y_all)
    if min_val_acc is not None:
        valid_sel &= sel_y_all > min_val_acc
        print(f"  {int(valid_sel.sum())} models above {min_val_acc}% validation accuracy")
    sel = select_n_front(x_all[valid_sel], sel_y_all[valid_sel], n_models)
    sel_pos = np.where(valid_sel)[0][sel]
    selected_mask = np.zeros(len(all_rows), dtype=bool)
    selected_mask[sel_pos] = True

    legend_elements = []
    legend_labels = []

    for sd_idx, sd in enumerate(studies_data):
        rows = [i for i, r in enumerate(all_rows) if r[0] == sd_idx]
        if not rows:
            continue

        # Models selected from the combined Pareto front that belong to this
        # study, plotted at their test-set accuracy
        sel_rows = [i for i in rows if selected_mask[i]]
        if sel_rows:
            xs = np.array([all_rows[i][1] for i in sel_rows])
            ys = np.array([all_rows[i][2] for i in sel_rows])
            ax.scatter(xs, ys, color=sd["color_par"], alpha=ALPHA_SELECTED,
                       s=70, marker=MARKER_SELECTED, edgecolors="white",
                       linewidths=0.6, zorder=4)
            for i in sel_rows:
                ax.annotate(str(all_rows[i][4]), (all_rows[i][1], all_rows[i][2]),
                            textcoords="offset points", xytext=(7, 7),
                            fontsize=7, fontweight="bold",
                            color=sd["color_par"], zorder=5)

            legend_elements.append(
                Line2D([0], [0], marker=MARKER_SELECTED, color="w",
                       markerfacecolor=sd["color_par"], markeredgecolor="white",
                       markeredgewidth=0.6, markersize=9, linestyle="none"))
            legend_labels.append(f"{sd['name']} (n={len(sel_rows)})")

    # ── Literature reference points (HAR studies only) ────────────────────
    # Logarithmic x-axis: parameters span orders of magnitude.  Set before
    # the literature annotations so get_xlim() below reflects log limits.
    ax.set_xscale("log")

    lit_handles = []
    lit_labels = []
    if any("har" in sd.get("study_name", "").lower()
           for sd in studies_data):
        metrics_plotted = set()
        for p in HAR_LITERATURE_POINTS:
            ax.scatter([p["params"]], [p["value"]], color=LIT_COLOR, s=45,
                       marker=LIT_MARKERS.get(p["metric"], "o"), zorder=4,
                       linewidths=1.0)
        x0, x1 = ax.get_xlim()
        for i, p in enumerate(HAR_LITERATURE_POINTS):
            # Annotate to the left for points on the right side of the plot,
            # and alternate above/below, to reduce label collisions. The
            # suffix carries the nuances from har-numbers-literature.md:
            # F1-score (not accuracy), compute cost, and free-text notes.
            right_side = p["params"] > np.sqrt(x0 * x1)
            dx = -7 if right_side else 7
            ha = "right" if right_side else "left"
            dy = 7 if i % 2 == 0 else -13
            parts = []
            if p["metric"] == "f1":
                parts.append("F1")
            if p.get("cost") is not None:
                parts.append(f"{p['cost']:g}M {p.get('cost_unit', 'MACs')}")
            if p.get("note"):
                parts.append(p["note"])
            suffix = f" ({', '.join(parts)})" if parts else ""
            ax.annotate(p["name"] + suffix, (p["params"], p["value"]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=6.5, color=LIT_COLOR, zorder=5)
            metric = p["metric"]
            if metric not in metrics_plotted:
                metrics_plotted.add(metric)
                lit_handles.append(
                    Line2D([0], [0], marker=LIT_MARKERS.get(metric, "o"),
                           color="w", markerfacecolor=LIT_COLOR,
                           markeredgecolor=LIT_COLOR, markersize=6,
                           linestyle="none", label=f"Literature ({metric})"))
                lit_labels.append(f"Literature ({metric})")

    ax.set_xlabel("Number of parameters", fontsize=11)
    ax.set_ylabel(accuracy_label or f"{accuracy_field.replace('_', ' ')} (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # x-axis in thousands (e.g. 5k = 5000 parameters); hide minor log-tick
    # labels to avoid clutter
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, pos: f"{v / 1000.0:g}k"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Frame y-axis around the data with a little padding, in percent
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0))

    ax.legend(handles=legend_elements + lit_handles,
              labels=legend_labels + lit_labels,
              framealpha=0.9, fontsize=9)

    savefig(fig, title, "param_accuracy")
