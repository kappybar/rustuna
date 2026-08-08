import tempfile

import optuna
import pytest
from optuna.storages import RDBStorage

import rustuna


def test_optimize_with_sqlite3() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"
        storage = rustuna.storages.SQLite3Storage(file_path, create_database=True)
        study = rustuna.create_study(storage=storage)

        def objective(trial: optuna.Trial | rustuna.Trial) -> float:
            x = trial.suggest_float("x", 1, 10, log=True)
            y = trial.suggest_int("y", -10, 10)
            trial.suggest_categorical("z", [True, False, "foo", 10])
            return x**2 + y

        study.optimize(objective, 10)
        assert len(study.trials) == 10


def test_use_optuna_db() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"

        # Create a database file
        RDBStorage(f"sqlite:///{file_path}")

        storage = rustuna.storages.SQLite3Storage(file_path)
        study = rustuna.create_study(storage=storage)

        def objective(trial: optuna.Trial | rustuna.Trial) -> float:
            x = trial.suggest_float("x", 1, 10, log=True)
            y = trial.suggest_int("y", -10, 10)
            trial.suggest_categorical("z", [True, False, "foo", 10])
            return x**2 + y

        study.optimize(objective, 10)
        assert len(study.trials) == 10


def test_sqlite3_storage_can_apply_discard() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/discarded.db"
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        study = rustuna.create_study(storage=storage, study_name="example")

        for _ in range(2):
            trial = study.ask()
            study.tell(trial.number, 1.0)

        trial_0_id = study.trials[0]._trial_id
        trial_1_id = study.trials[1]._trial_id
        storage.discard_trials([trial_0_id])
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            storage.get_trial(trial_0_id)

        # Resume
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        assert len(storage.get_trials(study._study_id)) == 1
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            storage.get_trial(trial_0_id)
        assert storage.get_trial(trial_1_id)._trial_id == trial_1_id
