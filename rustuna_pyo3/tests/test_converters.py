import optuna
import pytest

import rustuna
from rustuna import converter


@pytest.mark.parametrize(
    "optuna_direction",
    [optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MAXIMIZE],
)
def test_convert_direction(optuna_direction):
    rustuna_direction = converter.to_rustuna_direction(optuna_direction)
    restored = converter.to_optuna_direction(rustuna_direction)
    assert optuna_direction == restored


@pytest.mark.parametrize(
    "optuna_state",
    [
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.WAITING,
    ],
)
def test_convert_state(optuna_state):
    rustuna_state = converter.to_rustuna_state(optuna_state)
    restored = converter.to_optuna_state(rustuna_state)
    assert optuna_state == restored


@pytest.mark.parametrize(
    "optuna_distribution",
    [
        optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"]),
        optuna.distributions.FloatDistribution(low=0, high=1, step=0.1, log=False),
        optuna.distributions.IntDistribution(low=0, high=10, step=1, log=False),
    ],
)
def test_convert_distribution(optuna_distribution):
    rustuna_distribution = converter.to_rustuna_distribution(optuna_distribution)
    restored = converter.to_optuna_distribution(rustuna_distribution)
    assert optuna_distribution == restored
