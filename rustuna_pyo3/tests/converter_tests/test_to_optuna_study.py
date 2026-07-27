from __future__ import annotations

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import FrozenTrial, TrialState

import rustuna
from rustuna.converter import ToOptunaStudy


@pytest.fixture
def rustuna_study() -> rustuna.study.Study:
    return rustuna.create_study(
        storage=rustuna.storages.InMemoryStorage(),
        directions=["minimize"],
    )


@pytest.fixture
def study(rustuna_study: rustuna.study.Study) -> ToOptunaStudy:
    return ToOptunaStudy(rustuna_study)


def test_study_properties_and_user_attrs(
    study: ToOptunaStudy, rustuna_study: rustuna.study.Study
) -> None:
    rustuna_study.set_user_attr("raw", "value")

    assert study.study_name == rustuna_study.study_name
    assert study.directions == [optuna.study.StudyDirection.MINIMIZE]
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.user_attrs == {"raw": "value"}

    study.set_user_attr("json", {"answer": 42})
    assert study.user_attrs == {"raw": "value", "json": {"answer": 42}}


def test_get_trials_filters_states_and_respects_deepcopy(
    study: ToOptunaStudy,
) -> None:
    completed_trial = study.ask()
    study.tell(completed_trial, 1.0)
    running_trial = study.ask()

    assert [trial.number for trial in study.get_trials()] == [0, 1]
    assert [
        trial.number for trial in study.get_trials(states=(TrialState.COMPLETE,))
    ] == [0]
    assert [
        trial.number for trial in study.get_trials(states=(TrialState.RUNNING,))
    ] == [1]
    assert study.get_trials(states=[]) == []

    deep_copied = study.get_trials(deepcopy=True)
    assert all(isinstance(trial, FrozenTrial) for trial in deep_copied)
    deep_copied[0].params["new"] = 1
    assert "new" not in study.trials[0].params

    shallow = study.get_trials(deepcopy=False)
    assert shallow[0].number == completed_trial.number
    assert running_trial.number == 1


def test_optimize_updates_trial_data_and_invokes_callbacks(
    study: ToOptunaStudy,
) -> None:
    callback_trials: list[FrozenTrial] = []

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -1, 1)
        assert trial.params == {"x": x}
        assert trial.distributions == {"x": FloatDistribution(-1, 1)}
        trial.report(x**2, step=0)
        trial.set_user_attr("note", "from objective")
        assert trial.user_attrs == {"note": "from objective"}
        return x**2

    def callback(_: optuna.Study, trial: FrozenTrial) -> None:
        callback_trials.append(trial)

    study.optimize(objective, n_trials=3, callbacks=[callback])

    assert len(study.trials) == 3
    assert len(callback_trials) == 3
    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)
    assert all(trial.user_attrs == {"note": "from objective"} for trial in study.trials)
    assert all(trial.intermediate_values == {0: trial.value} for trial in study.trials)


@pytest.mark.parametrize("catch", [(), (ValueError,)])
def test_optimize_failure_and_catch(
    study: ToOptunaStudy,
    catch: tuple[type[Exception], ...],
) -> None:
    def objective(_: optuna.Trial) -> float:
        raise ValueError("expected failure")

    if catch:
        study.optimize(objective, n_trials=2, catch=catch)
    else:
        with pytest.raises(ValueError, match="expected failure"):
            study.optimize(objective, n_trials=1)

    assert all(trial.state == TrialState.FAIL for trial in study.trials)


def test_optimize_pruned(study: ToOptunaStudy) -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.report(1.0, step=0)
        raise optuna.TrialPruned

    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert all(trial.state == TrialState.PRUNED for trial in study.trials)
    assert all(trial.intermediate_values == {0: 1.0} for trial in study.trials)


def test_best_trial_properties(study: ToOptunaStudy) -> None:
    for value in [3.0, 1.0, 2.0]:
        study.tell(study.ask(), value)

    assert study.best_trial.number == 1
    assert study.best_value == 1.0
    assert study.best_params == {}
    assert study.best_trial == study.trials[1]


def test_multi_objective_properties() -> None:
    rustuna_study = rustuna.create_study(
        storage=rustuna.storages.InMemoryStorage(),
        directions=["minimize", "maximize"],
    )
    study = ToOptunaStudy(rustuna_study)

    assert study.directions == [
        optuna.study.StudyDirection.MINIMIZE,
        optuna.study.StudyDirection.MAXIMIZE,
    ]
    with pytest.raises(RuntimeError, match="single direction"):
        _ = study.direction

    for values in [[2.0, 2.0], [1.0, 1.0], [3.0, 1.0]]:
        study.tell(study.ask(), values)

    assert {tuple(trial.values or []) for trial in study.best_trials} == {
        (2.0, 2.0),
        (1.0, 1.0),
    }
    with pytest.raises(RuntimeError, match="single best trial"):
        _ = study.best_trial


def test_ask_and_tell_with_fixed_distributions(study: ToOptunaStudy) -> None:
    distributions = {
        "x": FloatDistribution(0, 1),
        "choice": CategoricalDistribution(["a", "b"]),
    }
    trial = study.ask(fixed_distributions=distributions)

    assert set(trial.params) == {"x", "choice"}
    assert 0 <= trial.params["x"] <= 1
    assert trial.params["choice"] in {"a", "b"}
    assert trial.distributions == distributions

    study.tell(trial, 1.0)
    assert study.trials[-1].params == trial.params
    assert study.trials[-1].distributions == distributions


def test_enqueue_trial(study: ToOptunaStudy) -> None:
    study.enqueue_trial({"x": 0.25}, user_attrs={"memo": "queued"})

    def objective(trial: optuna.Trial) -> float:
        assert trial.suggest_float("x", 0, 1) == 0.25
        assert trial.user_attrs == {"memo": "queued"}
        return 0.0

    study.optimize(objective, n_trials=1)

    assert study.trials[0].params == {"x": 0.25}
    assert study.trials[0].user_attrs == {"memo": "queued"}


def test_add_trial_and_add_trials(study: ToOptunaStudy) -> None:
    distribution = CategoricalDistribution(["a", "b"])
    trial = optuna.create_trial(
        params={"choice": "a"},
        distributions={"choice": distribution},
        value=1.0,
        user_attrs={"source": "single"},
    )
    study.add_trial(trial)
    study.add_trials(
        [
            optuna.create_trial(
                params={"choice": "b"},
                distributions={"choice": distribution},
                value=2.0,
                user_attrs={"source": "multiple"},
            )
        ]
    )

    assert len(study.trials) == 2
    assert study.trials[0].params == {"choice": "a"}
    assert study.trials[1].params == {"choice": "b"}
    assert [trial.user_attrs["source"] for trial in study.trials] == [
        "single",
        "multiple",
    ]


def test_stop(study: ToOptunaStudy) -> None:
    def objective(trial: optuna.Trial) -> float:
        if trial.number == 1:
            trial.study.stop()
        return float(trial.number)

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 2
