from __future__ import annotations

import optuna
import pytest

import rustuna
from rustuna.converter import ToOptunaStudy


def test_optuna_constraints_to_rustuna() -> None:
    rustuna_study = rustuna.create_study(
        storage=rustuna.storages.InMemoryStorage(),
        directions=["minimize"],
    )
    optuna_study = ToOptunaStudy(rustuna_study)

    trial = optuna_study.ask()
    trial.set_constraint("c0", 0.5)
    trial.set_constraint("c1", 10.0)
    frozen_trial = optuna_study.tell(
        trial, state=optuna.trial.TrialState.COMPLETE, values=1.0
    )

    persisted_trial = rustuna_study.trials[frozen_trial.number]
    assert frozen_trial.constraints == persisted_trial.constraints


def test_rustuna_constraints_to_optuna() -> None:
    rustuna_study = rustuna.create_study(
        storage=rustuna.storages.InMemoryStorage(),
        directions=["minimize"],
    )
    optuna_study = ToOptunaStudy(rustuna_study)

    trial = rustuna_study.ask()
    trial.set_constraints({"c0": 0.5, "c1": 10.0})
    persisted_trial = rustuna_study.tell(
        trial.number, state=rustuna.trial.TrialState.COMPLETE, values=1.0
    )

    frozen_trial = optuna_study.trials[persisted_trial.number]
    assert frozen_trial.constraints == persisted_trial.constraints
