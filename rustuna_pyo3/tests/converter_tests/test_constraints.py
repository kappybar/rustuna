from __future__ import annotations

import optuna

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


def test_optuna_constraints_to_rustuna_with_constraints_func() -> None:
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Binh and Korn function with constraints.
        x = trial.suggest_float("x", -15, 30)
        y = trial.suggest_float("y", -15, 30)
        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2

        # Modified constraints function.
        trial.set_user_attr("constraint", (1000 - v0,))
        return v0, v1

    def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
        return trial.user_attrs["constraint"]

    rustuna_study = rustuna.create_study(
        storage=rustuna.storages.InMemoryStorage(),
        directions=["minimize", "minimize"],
    )
    optuna_study = ToOptunaStudy(rustuna_study)
    optuna_study.sampler = optuna.samplers.NSGAIISampler(
        population_size=20,
        constraints_func=constraints,
    )
    optuna_study.optimize(objective, n_trials=50)

    for t in rustuna_study.trials[1:]:
        assert len(t.constraints) == 1
