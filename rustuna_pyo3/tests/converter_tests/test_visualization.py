# These tests are ported from the following file:
# https://github.com/optuna/optuna/blob/v4.9.0/tests/visualization_tests/test_visualizations.py

import warnings
from typing import Callable

import matplotlib.pyplot as plt
import optuna
import plotly.graph_objects as go
import pytest
from matplotlib.axes._axes import Axes
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
    plot_timeline,
)
from optuna.visualization.matplotlib import plot_contour as matplotlib_plot_contour
from optuna.visualization.matplotlib import plot_edf as matplotlib_plot_edf
from optuna.visualization.matplotlib import (
    plot_optimization_history as matplotlib_plot_optimization_history,
)
from optuna.visualization.matplotlib import (
    plot_parallel_coordinate as matplotlib_plot_parallel_coordinate,
)
from optuna.visualization.matplotlib import (
    plot_param_importances as matplotlib_plot_param_importances,
)
from optuna.visualization.matplotlib import plot_rank as matplotlib_plot_rank
from optuna.visualization.matplotlib import plot_slice as matplotlib_plot_slice
from optuna.visualization.matplotlib import plot_timeline as matplotlib_plot_timeline

import rustuna
from rustuna.converter import ToOptunaStudy

# https://github.com/optuna/optuna/blob/master/tests/visualization_tests/test_visualizations.py
parametrize_visualization_functions_for_single_objective = pytest.mark.parametrize(
    "plot_func",
    [
        plot_optimization_history,
        plot_edf,
        plot_contour,
        plot_parallel_coordinate,
        plot_rank,
        plot_slice,
        plot_timeline,
        plot_param_importances,
        matplotlib_plot_optimization_history,
        matplotlib_plot_edf,
        matplotlib_plot_contour,
        matplotlib_plot_parallel_coordinate,
        matplotlib_plot_rank,
        matplotlib_plot_slice,
        matplotlib_plot_timeline,
        matplotlib_plot_param_importances,
    ],
)


@parametrize_visualization_functions_for_single_objective
def test_visualization_for_single_objective(
    plot_func: Callable[[optuna.study.Study], go.Figure | Axes],
) -> None:
    def objective(trial: rustuna.Trial) -> float:
        category = trial.suggest_categorical("category", ["foo", "bar"])
        if category == "foo":
            return (trial.suggest_float("x1", 0, 10) - 2) ** 2
        else:
            return -((trial.suggest_float("x2", -10, 0) + 5) ** 2)

    rustuna_study = rustuna.create_study(sampler=rustuna.samplers.RandomSampler())
    rustuna_study.optimize(objective, n_trials=20)

    optuna_study = ToOptunaStudy(rustuna_study)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"Contour plot will not be displayed because .* cannot co-exist in `trial.params`\.",
            category=UserWarning,
        )
        fig = plot_func(optuna_study)  # Must not raise exception here.
    if isinstance(fig, Axes):
        plt.close()
