import re
import optuna
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances

from .common import savefig


# One entry per objective of the multi-objective studies in this project:
# (objective index, display name).
OBJECTIVES = [
    (0, "Accuracy"),
    (1, "Latency"),
]

# Canonical column order for the heatmaps: model architecture parameters
# first (matching the SEARCH_SPACE order in the config files), then the
# training hyperparameters.
PARAM_ORDER = ["d_model", "d_state", "d_conv", "expand", "lr", "optimizer"]


def _objective_importances(study, objective_id):
    """Importance of each hyperparameter for one objective.

    Uses the PedAnova evaluator on the raw objective value — the same
    computation as optuna-dashboard — so the numbers match what is shown
    there.  Note that PedAnova measures importance for *low* target values,
    so for the maximized Accuracy objective this answers "which parameters
    drive accuracy down" (e.g. lr leading to diverged training), not "which
    parameters drive accuracy up".
    """
    return get_param_importances(
        study,
        evaluator=PedAnovaImportanceEvaluator(),
        target=lambda t: t.values[objective_id],
    )


def _row_key(sd):
    """Order heatmap rows by dataset (HAR first), then architecture variant
    (single → bidir add → bidir mul), so the layout is stable regardless of
    the order/expansion of the config arguments."""
    name = sd["study_name"]
    dataset = 0 if sd.get("dataset", "").lower() == "har" else 1
    if "bidir-mul" in name:
        variant = 2
    elif "bidir" in name:
        variant = 1
    else:
        variant = 0
    return (dataset, variant, name)


def _row_label(sd):
    """Row label: the plot_description variant. The dataset is shown by the
    enclosing per-dataset subplot."""
    # Manual relabelling for the importance heatmaps only (config labels are
    # shared with other plot types): shortens bidirectional to bidir, which
    # also turns "Multi-layer bidirectional" into "Multi-layer bidir".
    label = re.sub(r"bidirectional", "bidir", sd["name"], flags=re.IGNORECASE)
    label = label[:1].upper() + label[1:]
    # Wrap the long labels onto two lines so they do not dominate the figure
    # width.
    if label.startswith("Multi-layer"):
        label = "Multi-layer\n" + label[len("Multi-layer"):].lstrip()
    elif label == "Single direction":
        label = "Single\ndirection"
    return label


def _dataset_groups(rows):
    """Split the (already dataset-sorted) rows into contiguous dataset groups:
    a list of (dataset, first_row, last_row) tuples in display order."""
    groups = []
    for i, (sd, _po, _nl) in enumerate(rows):
        dset = sd.get("dataset", "").lower()
        if groups and groups[-1][0] == dset:
            groups[-1] = (groups[-1][0], groups[-1][1], i)
        else:
            groups.append((dset, i, i))
    return groups


def _importance_cmap():
    """Importance colour scale: white -> pale yellow -> orange -> red.

    White for values near 0 (up to ~0.4 of the scale stays mostly pale), so
    tiny importances are barely tinted; the full red end is reached at 1.0.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        'white_yellow_orange_red',
        [(0.0, 'white'),
         (0.4, '#ffeda0'),
         (0.7, '#feb24c'),
         (1.0, '#bd0026')]
    )


def _study_searches_n_layers(study):
    """True if `n_layers` is an actual searched hyperparameter for this study.

    Single-layer studies fix n_layers at 1 (low=high, so no suggest_int call),
    it never appears in any trial's params, and is therefore not relevant to
    them.
    """
    return any(
        "n_layers" in t.params
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    )


def _draw_heatmap(ax, matrix, row_labels, col_labels, na_mask, show_x_labels=True):
    """One importance heatmap: rows are studies, columns are hyperparameters.

    All heatmaps share a 0–1 colour scale so colours mean the same thing
    across objectives (each row's importances sum to ~1). Cells marked True in
    `na_mask` (a param not searched by that study) are greyed out and show no
    number. Pass show_x_labels=False to drop the column labels (used for all
    panels except the bottom one, where labels would be duplicated).
    """
    im = ax.imshow(matrix, aspect="equal", cmap=_importance_cmap(), vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    if show_x_labels:
        ax.set_xticklabels(col_labels, rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=12)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if na_mask[r, c]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                           facecolor="#ececec", edgecolor="white",
                                           linewidth=1.2))
                continue
            if np.isnan(v):
                continue
            r_, g_, b_ = im.cmap(im.norm(v))[:3]
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            ax.text(c, r, f"{v:.2f}".lstrip("0") or "0", ha="center",
                    va="center", fontsize=11,
                    color="white" if lum < 0.55 else "black")
    return im


def create_param_importance_plot(studies_data):
    """
    Plot hyperparameter importances as one figure per objective (Accuracy,
    Latency): each figure holds two stacked heatmaps, one per dataset (HAR on
    top, KWS below), with the dataset name above each heatmap and column
    labels only on the bottom one. Rows are the studies, columns are the
    hyperparameters, cells are coloured by importance (PedAnova evaluator on
    the raw objective value, so the numbers agree with optuna-dashboard).

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name' (display name), 'study_name',
        'study' (loaded optuna.Study), ...
    """
    # Columns: canonical hyperparameter order plus any extras that appear in
    # some studies but are not listed in PARAM_ORDER.
    col_labels = list(PARAM_ORDER)
    rows = []  # (sd, [importance dict per objective], has_n_layers)
    row_labels = []
    for sd in sorted(studies_data, key=_row_key):
        n_completed = len(sd["study"].get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
        if n_completed < 2:
            print(f"  Not enough completed trials for {sd['study_name']} "
                  f"({n_completed}; need >= 2), skipping.")
            continue
        print(f"  {sd['study_name']}: {n_completed} completed trials")
        per_objective = []
        ok = True
        for objective_id, oname in OBJECTIVES:
            try:
                imp = _objective_importances(sd["study"], objective_id)
            except Exception as e:
                print(f"  Warning: hyperparameter importance for {oname} failed: {e}")
                ok = False
                break
            if not imp:
                print(f"  Warning: no hyperparameter importance for {oname}.")
                imp = {}
            per_objective.append(imp)
            for p in imp:
                if p not in col_labels:
                    col_labels.append(p)
        if not ok:
            continue
        rows.append((sd, per_objective, _study_searches_n_layers(sd["study"])))
        row_labels.append(_row_label(sd))

    if not rows:
        print("  No hyperparameter importance heatmaps created.")
        return False

    n_params = len(col_labels)
    matrix = np.zeros((len(rows), n_params))
    na_mask = np.zeros((len(rows), n_params), dtype=bool)
    if "n_layers" in col_labels:
        n_layer_col = col_labels.index("n_layers")
        for i, (sd, per_objective, has_n_layers) in enumerate(rows):
            if not has_n_layers:
                na_mask[i, n_layer_col] = True
    groups = _dataset_groups(rows)
    for objective_id, oname in OBJECTIVES:
        for i, (sd, per_objective, _has) in enumerate(rows):
            imp = per_objective[objective_id]
            for j, p in enumerate(col_labels):
                matrix[i, j] = np.nan if na_mask[i, j] else imp.get(p, 0.0)
        # One stacked heatmap per dataset instead of group labels inside a
        # single heatmap; the dataset name sits on top of each panel and the
        # column labels are only drawn on the bottom (last) panel. Equal
        # aspect (square cells) makes the required panel height proportional
        # to its row count; the figure height is derived so panels fill their
        # allocated width exactly (too tall would leave a blank gap between
        # the panels, too short a narrower, cropped heatmap).
        n_datasets = len(groups)
        # Target square cell size (inches): the width follows from it and the
        # height from the fill formula below, so the whole figure stays
        # page-sized while the cells remain square.
        cell_in = 0.45
        axes_frac = 0.705    # axes width / figure width (incl. colorbar)
        usable = 0.77        # fig.top - fig.bottom (default subplot margins)
        HS = 0.3            # vertical gap between panels (fraction of avg panel height)
        W = cell_in * n_params / axes_frac
        sum_rows = sum(end - start + 1 for _d, start, end in groups)
        t_needed = axes_frac * W * sum_rows / n_params
        H = t_needed * (1 + HS / n_datasets) / usable
        fig = plt.figure(figsize=(W, H))
        height_ratios = [end - start + 1 for _d, start, end in groups]
        gs = gridspec.GridSpec(n_datasets, 1, height_ratios=height_ratios,
                               wspace=0.05, hspace=HS)
        ims, axes = [], []
        for gi, (dset, start, end) in enumerate(groups):
            ax = fig.add_subplot(gs[gi, 0])
            axes.append(ax)
            last = gi == n_datasets - 1
            ims.append(_draw_heatmap(ax, matrix[start:end + 1],
                                     row_labels[start:end + 1], col_labels,
                                     na_mask[start:end + 1],
                                     show_x_labels=last))
            ax.set_title(dset.upper(), fontsize=13, fontweight="bold")
        cb = fig.colorbar(ims[-1], ax=axes, fraction=0.060, pad=0.04,
                         ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.tick_params(labelsize=10)
        savefig(fig, f"Hyperparameter Importances — {oname}", "param_importance",
                pad=1.5, tight_bbox=True)
        plt.close(fig)
    return True
