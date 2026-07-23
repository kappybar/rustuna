import sys

import pytest

import rustuna


def test_cmaes_sampler() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(
        trial.state == rustuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def test_cmaes_sampler_with_failed_trials() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        if trial.number % 3 == 0:
            raise ValueError("failed trial")
        return x**2 + y**2

    study.optimize(objective, n_trials=9, catch=ValueError)

    states = [trial.state for trial in study.trials]
    assert states.count(rustuna.trial.TrialState.COMPLETE) == 6
    assert states.count(rustuna.trial.TrialState.FAIL) == 3


def test_cmaes_sampler_ask_and_tell_with_abandoned_trial() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def suggest(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    abandoned = study.ask()
    suggest(abandoned)  # This trial is never told.

    for _ in range(6):
        trial = study.ask()
        study.tell(trial.number, values=suggest(trial))

    completed = study.get_trials(states=[rustuna.trial.TrialState.COMPLETE])
    assert len(completed) == 6


def test_cmaes_sampler_is_a_concrete_class() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1)
    assert isinstance(sampler, rustuna.samplers.CmaEsSampler)

    study = rustuna.create_study(sampler=sampler)
    assert isinstance(study.sampler, rustuna.samplers.CmaEsSampler)


def test_cmaes_sampler_missing_dependency_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cmaes", None)
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    with pytest.raises(ImportError, match="cmaes"):
        study.optimize(objective, n_trials=2)


def test_cmaes_sampler_with_single_value_parameters() -> None:
    """Test that CMAES correctly handles single-value parameters.

    Single-value parameters should be included in the search space and
    correctly transformed/untransformed by SearchSpaceTransform.
    """
    sampler = rustuna.samplers.CmaEsSampler(seed=42, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        # Normal parameters
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)

        # Single-value parameters (should be handled correctly)
        z = trial.suggest_float("z", 5.0, 5.0)  # single value: 5.0
        w = trial.suggest_int("w", 3, 3)  # single value: 3

        return x**2 + y**2 + z + w

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(
        trial.state == rustuna.trial.TrialState.COMPLETE for trial in study.trials
    )

    # Verify single-value parameters were always constant
    for trial in study.trials:
        assert trial.params["z"] == 5.0, f"z should always be 5.0, got {trial.params['z']}"
        assert trial.params["w"] == 3, f"w should always be 3, got {trial.params['w']}"
