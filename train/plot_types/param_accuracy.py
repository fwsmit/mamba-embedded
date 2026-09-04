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
FIGSIZE = (10.5, 9.5)
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 9
LIT_COLORS = plt.get_cmap("tab10").colors
MODEL_MARKERS = ("o", "^", "s", "P", "v", "<", ">", "h", "p", "*", "X")
OURS_COLOR = "#2E7D32"


# ── Literature reference points (overlaid for HAR/KWS studies) ─────────────
# Sources collected in har-numbers-literature.md and kws-numbers-literature.md.
# To add a new point, append a
# dict with these keys:
#   name      : short label used in the legend
#   params    : number of parameters (scalar)
#   value     : reported accuracy in percent (or F1-score, see metric)
#   metric    : "accuracy" or "f1" — f1 points use open markers in the plot
#   source    : free-text citation (arXiv id / paper title)
#   group     : manually assigned model family used for markers and lines
#   cost      : optional compute cost in millions (MACs or FLOPs, see cost_unit)
#   cost_unit : "MACs" or "FLOPs" (only meaningful with ``cost``)
#   note      : optional free-text nuance (e.g. "average across datasets")
HAR_LITERATURE_POINTS = [
    dict(name="Novac et al.", params=3958, value=92.41, metric="accuracy",
         source="arXiv:2105.13331", group="Novac et al."),
    dict(name="MicrobiconvLSTM", params=11400, value=93.41, metric="accuracy",
         source="arXiv:2602.06523", group="MicrobiconvLSTM", cost=0.42, cost_unit="MACs",
         note="average"),
    dict(name="Machar", params=67380, value=99.32, metric="accuracy",
         source="arXiv:2602.06523", group="Machar", cost=10.37, cost_unit="MFLOPs"),
    dict(name="Crossover-BiDir-BabyMamba", params=27000, value=95.10,
         metric="f1", source="BabyMamba-HAR", group="BabyMamba-HAR",
         cost=2.21, cost_unit="MACs"),
    dict(name="CI-BabyMamba-HAR", params=28000, value=85.80, metric="f1",
         source="BabyMamba-HAR", group="BabyMamba-HAR", cost=50.92,
         cost_unit="MACs"),
    dict(name="TinyHAR", params=55000, value=96.53, metric="f1",
         source="BabyMamba-HAR", group="BabyMamba-HAR", cost=9.29,
         cost_unit="MACs"),
    dict(name="TinierHAR", params=33000, value=96.37, metric="f1",
         source="BabyMamba-HAR", group="BabyMamba-HAR", cost=1.73,
         cost_unit="MACs"),
    dict(name="DeepConvLSTM", params=136000, value=93.53, metric="f1",
         source="BabyMamba-HAR", group="BabyMamba-HAR", cost=15.51,
         cost_unit="MACs"),
    dict(name="MambaLite-Micro", params=37100, value=92.7, metric="accuracy",
         source="MambaLite-Micro", group="MambaLite-Micro", cost=123.4,
         cost_unit="ms"),
    dict(name="HARMamba", params=388300, value=97.01, metric="f1",
         source="HARMamba", group="HARMamba", cost=11.07,
         cost_unit="MFLOPs"),
]

# Sources collected in kws-numbers-literature.md.  The LMU models report their
# model size in kbits rather than a parameter count; they are converted to
# parameters here assuming int8 storage (1 byte = 1 param, kbits / 8), which
# is noted in the legend.  `cost` carries each TinySpeech model's
# compute in millions of Mult-Adds.
KWS_LITERATURE_POINTS = [
    dict(name="MambaLite-Micro", params=35978, value=92.5, metric="accuracy",
         source="MambaLite-Micro", group="MambaLite-Micro", cost=1133.6,
         cost_unit="ms"),
    dict(name="TinySpeech-X", params=10800, value=94.6, metric="accuracy",
         source="arXiv:2008.04245", group="TinySpeech", cost=10.9,
         cost_unit="Mult-Adds"),
    dict(name="TinySpeech-Y", params=6100, value=93.6, metric="accuracy",
         source="arXiv:2008.04245", group="TinySpeech", cost=6.5,
         cost_unit="Mult-Adds"),
    dict(name="TinySpeech-Z", params=2700, value=92.4, metric="accuracy",
         source="arXiv:2008.04245", group="TinySpeech", cost=2.6,
         cost_unit="Mult-Adds"),
    dict(name="LMU-1", params=210375, value=96.9, metric="accuracy",
         source="arXiv:2009.04465", group="LMU", note="model size, int8 params"),
    dict(name="LMU-2", params=45125, value=95.9, metric="accuracy",
         source="arXiv:2009.04465", group="LMU", note="model size, int8 params"),
    dict(name="LMU-3", params=13125, value=95.0, metric="accuracy",
         source="arXiv:2009.04465", group="LMU", note="model size, int8 params"),
    dict(name="LMU-4", params=6125, value=92.7, metric="accuracy",
         source="arXiv:2009.04465", group="LMU", note="model size, int8 params"),
    # Further points from kws-numbers-literature-full-report.md.
    dict(name="TinySpeech-M", params=4700, value=91.9, metric="accuracy",
         source="arXiv:2008.04245", group="TinySpeech", cost=4.4,
         cost_unit="Mult-Adds"),
    dict(name="MicroCNN", params=4200, value=93.22, metric="accuracy",
         source="arXiv:2511.07821", group="MicroCNN"),
    dict(name="DS-CNN-S", params=4400, value=91.5, metric="accuracy",
         source="ICASSP 2019", group="DS-CNN", cost=5.4,
         cost_unit="Mult-Adds", note="~91-92%"),
    dict(name="TC-ResNet8-0.25", params=5600, value=90.5, metric="accuracy",
         source="InterSpeech 2019", group="TC-ResNet8-0.25"),
    dict(name="Res8-narrow", params=20000, value=90.1, metric="accuracy",
         source="InterSpeech 2019", group="Res8-narrow", cost=143.2,
         cost_unit="Mult-Adds"),
    dict(name="TENet-6-narrow", params=17000, value=96.0, metric="accuracy",
         source="Interspeech 2020", group="TENet", cost=0.553,
         cost_unit="Mult-Adds"),
    dict(name="TKWS-3", params=14400, value=92.4, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="TKWS"),
    dict(name="TKWS-2", params=4600, value=88.8, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="TKWS"),
    dict(name="TENet6-N", params=17100, value=91.8, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="TENet"),
    dict(name="TENet6", params=54200, value=92.9, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="TENet"),
    dict(name="LicoNet-S", params=17400, value=93.6, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="LicoNet-S"),
    dict(name="DS-CNN", params=46500, value=91.2, metric="accuracy",
         source="bartoliEndtoEndEfficiencyKeyword2025", group="DS-CNN"),
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


def literature_group(point):
    """Return the manually assigned model group for a literature point."""
    return point["group"]


def create_param_accuracy_plot(studies_data, title, n_models=10,
                               accuracy_field="test_quantized_accuracy",
                               selection_accuracy_field=None,
                               accuracy_label=None,
                               min_val_acc=None):
    """
    Scatter plot of number of parameters vs accuracy for the N most
    parameter-efficient models, selected from the combined Pareto front
    across all studies (max validation accuracy, min parameters) and plotted
    with model-family markers and connecting lines at their test-set
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
    overlaid with one marker shape per model family. Variants in a family are
    connected by a line and all paper names are collected in the legend below
    the plot.
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
    ours_rows = []

    for sd_idx, sd in enumerate(studies_data):
        rows = [i for i, r in enumerate(all_rows) if r[0] == sd_idx]
        if not rows:
            continue
        ours_rows.extend(i for i in rows if selected_mask[i])

    if ours_rows:
        xs = np.array([all_rows[i][1] for i in ours_rows])
        ys = np.array([all_rows[i][2] for i in ours_rows])
        order = np.argsort(xs)
        if len(ours_rows) > 1:
            ax.plot(xs[order], ys[order], color=OURS_COLOR,
                    linewidth=1.8, alpha=0.85, zorder=2)
        ax.scatter(xs, ys, color=OURS_COLOR, alpha=ALPHA_SELECTED,
                   s=180, marker="D", edgecolors="white",
                   linewidths=1.2, zorder=4)
        legend_elements.append(
            Line2D([0], [0], marker="D", color=OURS_COLOR,
                   markerfacecolor=OURS_COLOR, markeredgecolor="white",
                   markeredgewidth=0.6, markersize=8, linewidth=1.5))
        legend_labels.append(f"Ours (n={len(ours_rows)})")

    # ── Literature reference points ───────────────────────────────────────
    ax.set_xscale("log")

    lit_points = []
    lit_datasets = set()
    for sd in studies_data:
        for dataset, points in (("har", HAR_LITERATURE_POINTS),
                                ("kws", KWS_LITERATURE_POINTS)):
            if dataset in sd.get("study_name", "").lower() and dataset not in lit_datasets:
                lit_datasets.add(dataset)
                lit_points += points

    if lit_points:
        families = {}
        for p in lit_points:
            families.setdefault(literature_group(p), []).append(p)
        family_colors = {
            family: LIT_COLORS[i % len(LIT_COLORS)]
            for i, family in enumerate(families)
        }
        family_markers = {
            family: MODEL_MARKERS[i % len(MODEL_MARKERS)]
            for i, family in enumerate(families)
        }

        for family, points in families.items():
            points = sorted(points, key=lambda p: p["params"])
            color = family_colors[family]
            if len(points) > 1:
                ax.plot([p["params"] for p in points],
                        [p["value"] for p in points], color=color,
                        linewidth=1.5, alpha=0.8, zorder=2)
            for p in points:
                marker = family_markers[family]
                filled = p["metric"] != "f1"
                ax.scatter([p["params"]], [p["value"]], color=color,
                           facecolors=color if filled else "none",
                           edgecolors=color, s=95, marker=marker,
                           linewidths=1.4, zorder=4)

        # One entry per paper, while retaining all model names when a paper
        # reports several variants.
        papers = {}
        for p in lit_points:
            key = (p["source"], literature_group(p))
            papers.setdefault(key, []).append(p)
        for (source, family), points in papers.items():
            names = "/".join(p["name"] for p in points)
            metric = "F1" if all(p["metric"] == "f1" for p in points) else "accuracy"
            notes = sorted({p["note"] for p in points if p.get("note")})
            details = ", ".join([metric] + notes)
            marker = family_markers[family]
            color = family_colors[family]
            legend_elements.append(
                Line2D([0], [0], marker=marker, color=color,
                       markerfacecolor="none" if metric == "F1" else color,
                       markeredgecolor=color, markersize=7, linewidth=1.3))
            legend_labels.append(f"{names} ({source}, {details})")

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

    ax.legend(handles=legend_elements, labels=legend_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, framealpha=0.95, fontsize=LEGEND_SIZE,
              borderpad=0.7, labelspacing=0.65, columnspacing=1.0,
              handletextpad=0.5)
    fig.subplots_adjust(bottom=0.30)

    savefig(fig, title, "param_accuracy", dpi=300, tight=False, tight_bbox=True)
