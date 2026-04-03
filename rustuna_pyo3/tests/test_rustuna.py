import pytest

import rustuna


def test_optimize():
    study = rustuna.create_study()

    def objective(trial: rustuna.Trial):
        x = trial.suggest_float("x", -5.0, 5)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert -5.0 <= x <= 5.0
        assert 0 <= y <= 10
        assert z in ["foo", "bar"]
        trial.set_user_attr("key", "value")
        assert trial.user_attrs == {"key": "value"}
        return x * 2 + y

    study.optimize(objective, n_trials=10)

    assert len(study.best_trial.internal_params) == 3
    assert len(study.best_trial.distributions) == 3
    assert len(study.best_trial.params) == 3
    assert study.best_trial.value is not None


def test_trial_storage_inside_objective():
    storage = rustuna.Storage.in_memory()
    study = rustuna.create_study(storage=storage)

    def objective(trial: rustuna.Trial):
        assert trial.storage is storage
        assert len(trial.storage.get_studies()) == 1
        return trial.suggest_float("x", -1.0, 1.0)

    study.optimize(objective, n_trials=1)


def test_study_get_trials_filters_by_states():
    study = rustuna.create_study()
    failed_trial = study.ask()
    completed_trial = study.ask()
    study.tell(failed_trial.number, state=rustuna.TrialState.FAIL)
    study.tell(completed_trial.number, values=1.0)

    filtered = study.get_trials(
        states=[rustuna.TrialState.FAIL, rustuna.TrialState.COMPLETE]
    )

    assert len(filtered) == 2
    assert [trial.state for trial in filtered] == [
        rustuna.TrialState.FAIL,
        rustuna.TrialState.COMPLETE,
    ]


def test_optimize_multi_objective():
    study = rustuna.create_study(directions=["minimize", "minimize"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_int("y", 0, 3)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert 0 <= x <= 5.0
        assert 0 <= y <= 3
        assert z in ["foo", "bar"]

        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study.optimize(objective, n_trials=10)

    trial = study.trials[-1]
    assert len(trial.internal_params) == 3
    assert len(trial.distributions) == 3
    assert len(trial.params) == 3
    assert len(trial.values) == 2


def test_study():
    study = rustuna.create_study(study_name="example")

    assert study.name == "example"

    study.set_user_attr("key", "value")
    assert study.user_attrs == {"key": "value"}

    assert len(study.storage.get_studies()) == 1


def test_suggest_categorical():
    study = rustuna.create_study()

    def objective(trial):
        x = trial.suggest_categorical("x", ["foo", 1, 1.0, None, True, False])

        assert x in ["foo", 1, 1.0, None, True, False]
        return 0.0

    study.optimize(objective, n_trials=10)


def test_fanova():
    study = rustuna.create_study()

    def objective(trial):
        x = trial.suggest_float("x", -5.0, 5)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert -5.0 <= x <= 5.0
        assert 0 <= y <= 10
        assert z in ["foo", "bar"]
        return x * 2 + y

    study.optimize(objective, n_trials=10)
    rustuna.get_param_importance(study)


def test_ask():
    study = rustuna.create_study(study_name="example")
    trial = study.ask()

    # Suggest APIs
    x = trial.suggest_float("x", 0, 1)
    assert 0 <= x <= 1
    y = trial.suggest_int("y", 0, 10)
    assert 0 <= y <= 10
    z = trial.suggest_categorical("z", ["foo", "bar"])
    assert z in ["foo", "bar"]

    # User Attrs
    x = trial.suggest_float("x", 0, 1)
    trial.set_user_attr("foo", "bar")

    # Tell
    study.tell(trial.number, 1.0)


def test_trial_state():
    state = rustuna.TrialState.COMPLETE
    assert state.is_finished()


def test_persisted_trial():
    trial = rustuna.PersistedTrial(
        trial_id=2,
        study_id=1,
        number=2,
        state=rustuna.TrialState.COMPLETE,
        values=[0.5],
    )
    assert trial.study_id == 1
    assert trial.number == 2
    assert trial.state.is_finished()
    assert trial.values == [0.5]

    pytest.raises(
        ValueError,
        lambda: rustuna.PersistedTrial(2, 1, 2, state=rustuna.TrialState.COMPLETE),
    )


def test_sample():
    samplers = [
        rustuna.Sampler.tpe(),
        rustuna.Sampler.random(),
    ]
    for sampler in samplers:
        storage = rustuna.Storage.in_memory()
        study = rustuna.create_study(sampler=sampler, storage=storage)
        trial = study.ask()
        ctx = rustuna.SamplerContext(
            study_id=study.id,
            trial_number=trial.number,
            trial_id=trial.id,
            directions=study.directions,
        )
        value = sampler.sample_independent(
            ctx, storage, "x", rustuna.Distribution.float(0, 1)
        )
        assert 0 <= value <= 1


def test_storage():
    storage = rustuna.Storage.in_memory()
    study = storage.create_new_study("example", [rustuna.StudyDirection.MINIMIZE])


def test_get_pareto_front():
    study = rustuna.create_study(directions=["minimize", "minimize"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_int("y", 0, 3)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert 0 <= x <= 5.0
        assert 0 <= y <= 3
        assert z in ["foo", "bar"]

        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study.optimize(objective, n_trials=10)

    trials = study.best_trials
    assert len(trials) > 0
    assert len(trials) <= 10
