import optuna
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    dataset = "har" if "har" in name else "kws"
    if "bidir-mul" in name:
        variant = 2
    elif "bidir" in name:
        variant = 1
    else:
        variant = 0
    return (dataset, variant, name)


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


def _draw_heatmap(ax, matrix, row_labels, col_labels):
    """One importance heatmap: rows are studies, columns are hyperparameters.

    All heatmaps share a 0–1 colour scale so colours mean the same thing
    across objectives (each row's importances sum to ~1).
    """
    im = ax.imshow(matrix, aspect="auto", cmap=_importance_cmap(), vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if np.isnan(v):
                continue
            r_, g_, b_ = im.cmap(im.norm(v))[:3]
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if lum < 0.55 else "black")
    return im


def create_param_importance_plot(studies_data):
    """
    Plot hyperparameter importances as one heatmap per objective (Accuracy,
    Latency): rows are the studies, columns are the hyperparameters, cells
    are coloured by importance (PedAnova evaluator on the raw objective
    value, so the numbers agree with optuna-dashboard).

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name' (display name), 'study_name',
        'study' (loaded optuna.Study), ...
    """
    # Columns: canonical hyperparameter order plus any extras that appear in
    # some studies but are not listed in PARAM_ORDER.
    col_labels = list(PARAM_ORDER)
    rows = []  # (sd, [importance dict per objective])
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
        rows.append((sd, per_objective))
        row_labels.append(sd["study_name"])

    if not rows:
        print("  No hyperparameter importance heatmaps created.")
        return False

    n_params = len(col_labels)
    matrix = np.zeros((len(rows), n_params))
    for objective_id, oname in OBJECTIVES:
        for i, (sd, per_objective) in enumerate(rows):
            imp = per_objective[objective_id]
            for j, p in enumerate(col_labels):
                matrix[i, j] = imp.get(p, 0.0)
        fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * n_params + 3.0),
                                        0.6 * len(rows) + 2.6))
        im = _draw_heatmap(ax, matrix, row_labels, col_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Importance")
        savefig(fig, f"Hyperparameter Importances — {oname}", "param_importance")
        plt.close(fig)
    return True
