import optuna
import numpy as np
import matplotlib.pyplot as plt

from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances

from .common import savefig


# One entry per objective of the multi-objective studies in this project:
# (objective index, display name, direction, bar colour).
OBJECTIVES = [
    (0, "Accuracy", "maximize", "#4C9BE8"),
    (1, "Latency", "minimize", "#E8834C"),
]


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


def _draw_panel(ax, importance, objective_name, direction, color):
    """Horizontal bar chart, most important parameter at the top."""
    params = list(importance)  # get_param_importances returns descending order
    values = [importance[p] for p in params]
    y = np.arange(len(params))
    ax.barh(y, values, height=0.6, color=color, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=10)
    ax.set_xlabel("Hyperparameter Importance", fontsize=11)
    ax.set_title(f"{objective_name} ({direction})", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for yi, v in zip(y, values):
        ax.text(v, yi, f"{v:.2f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, max(values) * 1.15)


def create_param_importance_plot(studies_data):
    """
    Plot hyperparameter importances for each study, one figure per study
    with one panel per objective.

    Importance values are computed exactly like optuna-dashboard (PedAnova
    evaluator on the raw objective value), so the numbers agree with the
    dashboard; only the rendering differs: each objective gets its own panel
    with bars ordered from most to least important, instead of optuna's
    merged single-axis layout where the shared y-tick labels are attached to
    only one of the two bar series.

    Parameters
    ----------
    studies_data : list of dict
        Each dict has keys: 'name' (display name), 'study_name',
        'study' (loaded optuna.Study), ...
    """
    created = False
    for sd in studies_data:
        n_completed = len(sd["study"].get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
        if n_completed < 2:
            print(f"  Not enough completed trials for {sd['study_name']} "
                  f"({n_completed}; need >= 2), skipping.")
            continue

        if sd["name"] == sd["study_name"]:
            fig_title = f"{sd['study_name']} Hyperparameter Importances"
        else:
            fig_title = f"{sd['study_name']} ({sd['name']}) Hyperparameter Importances"
        print(f"  {sd['study_name']}: {n_completed} completed trials")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, (objective_id, oname, odir, color) in zip(axes, OBJECTIVES):
            try:
                imp = _objective_importances(sd["study"], objective_id)
            except Exception as e:
                print(f"  Warning: hyperparameter importance for {oname} failed: {e}")
                ax.set_visible(False)
                continue
            if not imp:
                print(f"  Warning: no hyperparameter importance for {oname}.")
                ax.set_visible(False)
                continue
            _draw_panel(ax, imp, oname, odir, color)

        fig.suptitle(fig_title, y=0.98, fontsize=14, fontweight="bold")
        savefig(fig, fig_title, "param_importance")
        created = True

    if not created:
        print("  No hyperparameter importance plots created.")
    return created
