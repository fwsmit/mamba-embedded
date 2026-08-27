"""
Parameter-efficiency scatter plot: number of parameters (x) vs test-set
accuracy (y), with the N best models selected from the Pareto front of
(min parameters, max validation-set accuracy) and highlighted.

Usage (via plot_arch_search.py):
  python -m train.plot_arch_search --plot param_accuracy \\
      --n-models 10 --size 8 --quantization percent config/har/*.yaml

When the studies target the HAR or KWS dataset, reference points from the
literature (har-numbers-literature.md and kws-numbers-literature.md, see
HAR_LITERATURE_POINTS / KWS_LITERATURE_POINTS below) are overlaid on the same
axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

from .common import savefig

ALPHA_SELECTED = 0.95
MARKER_SELECTED = "D"
FIGSIZE = (10.5, 8)
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
TITLE_SIZE = 16
LEGEND_SIZE = 11
ANNOTATION_SIZE = 10
LITERATURE_ANNOTATION_SIZE = 9


# ── Literature reference points (overlaid for HAR/KWS studies) ─────────────
# Sources collected in har-numbers-literature.md and kws-numbers-literature.md.
# To add a new point, append a
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
LIT_ANNOTATION_OFFSETS = {
    "TinierHAR": (-7, 10, "right"),
    "TinyHAR": (7, -13, "left"),
    "TinySpeech-X": (7, -13, "left"),
    "LMU-4": (7, 18, "left"),
    "TinySpeech-Y": (7, -22, "left"),
    "MambaLite-Micro": (7, -18, "left"),
    "MicrobiconvLSTM": (7, -14, "left"),
}
HAR_LITERATURE_POINTS = [
    dict(name="Novac et al.", params=3958, value=92.41, metric="accuracy",
         source="arXiv:2105.13331"),
    dict(name="MicrobiconvLSTM", params=11400, value=93.41, metric="accuracy",
         source="arXiv:2602.06523", cost=0.42, cost_unit="MACs",
         note="average"),
    dict(name="Machar", params=67380, value=99.32, metric="accuracy",
         source="arXiv:2602.06523", cost=10.37, cost_unit="MFLOPs"),
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
    dict(name="MambaLite-Micro", params=37100, value=92.7, metric="accuracy",
         source="MambaLite-Micro", cost=123.4, cost_unit="ms"),
    dict(name="HARMamba", params=388300, value=97.01, metric="f1",
         source="HARMamba", cost=11.07, cost_unit="MFLOPs"),
]

# Sources collected in kws-numbers-literature.md.  The LMU models report their
# model size in kbits rather than a parameter count; they are converted to
# parameters here assuming int8 storage (1 byte = 1 param, kbits / 8), which
# is noted on their annotations.  `cost` carries each TinySpeech model's
# compute in millions of Mult-Adds.
KWS_LITERATURE_POINTS = [
    dict(name="TinySpeech-X", params=10800, value=94.6, metric="accuracy",
         source="arXiv:2008.04245", cost=10.9, cost_unit="Mult-Adds"),
    dict(name="TinySpeech-Y", params=6100, value=93.6, metric="accuracy",
         source="arXiv:2008.04245", cost=6.5, cost_unit="Mult-Adds"),
    dict(name="TinySpeech-Z", params=2700, value=92.4, metric="accuracy",
         source="arXiv:2008.04245", cost=2.6, cost_unit="Mult-Adds"),
    dict(name="LMU-1", params=210375, value=96.9, metric="accuracy",
         source="arXiv:2009.04465", note="model size, int8 params"),
    dict(name="LMU-2", params=45125, value=95.9, metric="accuracy",
         source="arXiv:2009.04465", note="model size, int8 params"),
    dict(name="LMU-3", params=13125, value=95.0, metric="accuracy",
         source="arXiv:2009.04465", note="model size, int8 params"),
    dict(name="LMU-4", params=6125, value=92.7, metric="accuracy",
         source="arXiv:2009.04465", note="model size, int8 params"),
    # Further points from kws-numbers-literature-full-report.md.
    dict(name="TinySpeech-M", params=4700, value=91.9, metric="accuracy",
         source="arXiv:2008.04245", cost=4.4, cost_unit="Mult-Adds"),
    dict(name="MicroCNN", params=4200, value=93.22, metric="accuracy",
         source="arXiv:2511.07821"),
    dict(name="DS-CNN-S", params=4400, value=91.5, metric="accuracy",
         source="ICASSP 2019", cost=5.4, cost_unit="Mult-Adds",
         note="~91-92%"),
    dict(name="TC-ResNet8-0.25", params=5600, value=90.5, metric="accuracy",
         source="InterSpeech 2019"),
    dict(name="Res8-narrow", params=20000, value=90.1, metric="accuracy",
         source="InterSpeech 2019", cost=143.2, cost_unit="Mult-Adds"),
    dict(name="TENet-6-narrow", params=17000, value=96.0, metric="accuracy",
         source="Interspeech 2020", cost=0.553, cost_unit="Mult-Adds"),
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

    When any study targets the HAR or KWS dataset, reference points from the
    literature (see HAR_LITERATURE_POINTS / KWS_LITERATURE_POINTS) are
    overlaid as black markers: accuracy values as crosses, F1-scores as
    pentagons annotated with an "(F1)" suffix.
    """
    sel_field = selection_accuracy_field or accuracy_field

    print("Selecting from", sel_field)
    fig, ax = plt.subplots(figsize=FIGSIZE)

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

    # Print the selected trials to the terminal.
    if selected_mask.any():
        dmodels = {}
        for sd_idx_ in {r[0] for r in all_rows}:
            try:
                dmodels[sd_idx_] = {t.number: t.params.get("d_model")
                                    for t in studies_data[sd_idx_]["study"].trials}
            except Exception:
                dmodels[sd_idx_] = {}
        print(f"  Selected {int(selected_mask.sum())} model(s) from the Pareto front:")
        print(f"    {'trial':>5} {'study':<22} {'d_model':>7} {'params':>9} "
              f"{'val_acc':>8} {'test_acc':>8}")
        for i, r in enumerate(all_rows):
            if not selected_mask[i]:
                continue
            sd_idx_, npar_, test_acc_, sel_acc_, trial_ = r
            dn = dmodels.get(sd_idx_, {}).get(trial_)
            dn_s = f"{dn}" if dn is not None else "-"
            print(f"    {trial_:>5} {studies_data[sd_idx_]['name']:<22} "
                  f"{dn_s:>7} {npar_:>9} {sel_acc_:>7.2f}% {test_acc_:>7.2f}%")
        print()
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
                       s=260, marker=MARKER_SELECTED, edgecolors="white",
                       linewidths=1.2, zorder=4)
            legend_elements.append(
                Line2D([0], [0], marker=MARKER_SELECTED, color="w",
                       markerfacecolor=sd["color_par"], markeredgecolor="white",
                       markeredgewidth=0.6, markersize=9, linestyle="none"))
            legend_labels.append(f"{sd['name']} (n={len(sel_rows)})")

    # ── Literature reference points (HAR studies only) ────────────────────
    # Logarithmic x-axis: parameters span orders of magnitude.  Set before
    # the literature annotations so get_xlim() below reflects log limits.
    ax.set_xscale("log")

    lit_points = []
    lit_datasets = set()
    for sd in studies_data:
        for dataset, points in (("har", HAR_LITERATURE_POINTS),
                                ("kws", KWS_LITERATURE_POINTS)):
            if dataset in sd.get("study_name", "").lower() and dataset not in lit_datasets:
                lit_datasets.add(dataset)
                lit_points += points

    lit_handles = []
    lit_labels = []
    if lit_points:
        metrics_plotted = set()
        for p in lit_points:
            ax.scatter([p["params"]], [p["value"]], color=LIT_COLOR, s=160,
                       marker=LIT_MARKERS.get(p["metric"], "o"), zorder=4,
                       linewidths=1.2)
        x0, x1 = ax.get_xlim()
        for i, p in enumerate(lit_points):
            # Annotate to the left for points on the right side of the plot,
            # and alternate above/below, to reduce label collisions. The
            # suffix carries the nuances from har-numbers-literature.md and
            # kws-numbers-literature.md: F1-score (not accuracy), compute
            # cost, and free-text notes.
            right_side = p["params"] > np.sqrt(x0 * x1)
            dx = -7 if right_side else 7
            ha = "right" if right_side else "left"
            dy = 7 if i % 2 == 0 else -13
            if p["name"] in LIT_ANNOTATION_OFFSETS:
                dx, dy, ha = LIT_ANNOTATION_OFFSETS[p["name"]]
            parts = []
            if p["metric"] == "f1":
                parts.append("F1")
            if p.get("note"):
                parts.append(p["note"])
            suffix = f" ({', '.join(parts)})" if parts else ""
            ax.annotate(p["name"] + suffix, (p["params"], p["value"]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=LITERATURE_ANNOTATION_SIZE, color=LIT_COLOR,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                                  pad=1.5), zorder=5)
            metric = p["metric"]
            if metric not in metrics_plotted:
                metrics_plotted.add(metric)
                lit_handles.append(
                    Line2D([0], [0], marker=LIT_MARKERS.get(metric, "o"),
                           color="w", markerfacecolor=LIT_COLOR,
                           markeredgecolor=LIT_COLOR, markersize=12,
                           linestyle="none", label=f"Literature ({metric})"))
                lit_labels.append(f"Literature ({metric})")

    ax.set_xlabel("Number of parameters", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(accuracy_label or f"{accuracy_field.replace('_', ' ')} (%)",
                  fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
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
              framealpha=0.9, fontsize=LEGEND_SIZE,
              borderpad=0.7, labelspacing=0.5)

    savefig(fig, title, "param_accuracy", dpi=300)
