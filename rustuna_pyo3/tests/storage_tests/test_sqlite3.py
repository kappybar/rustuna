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


def test_optuna_can_use_a_database_created_by_rustuna() -> None:
    # Rustuna adds a `discarded_at` column to the trials table, so make sure Optuna can still
    # read and extend a database that Rustuna initialized.
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        study = rustuna.create_study(storage=storage, study_name="example")
        for _ in range(3):
            trial = study.ask()
            study.tell(trial.number, 1.0)
        trial_ids = [t._trial_id for t in study.trials]
        storage.discard_trials([trial_ids[0]])

        optuna_study = optuna.load_study(
            study_name="example", storage=RDBStorage(f"sqlite:///{file_path}")
        )
        assert len(optuna_study.trials) == 3
        optuna_study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
        assert len(optuna_study.trials) == 5

        # The discard survives the writes made through Optuna.
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        assert len(storage.get_trials(study._study_id)) == 4
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            storage.get_trial(trial_ids[0])


def test_sqlite3_storage_discards_are_recorded_without_apply_discard() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/discarded.db"
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=False)
        study = rustuna.create_study(storage=storage, study_name="example")
        for _ in range(2):
            trial = study.ask()
            study.tell(trial.number, 1.0)
        trial_ids = [t._trial_id for t in study.trials]

        # As with JournalStorage, the discard is written even though this storage does not
        # apply it when reading.
        storage.discard_trials([trial_ids[0]])
        assert not storage.may_omit_trials()
        assert storage.get_trial(trial_ids[0])._trial_id == trial_ids[0]

        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        assert len(storage.get_trials(study._study_id)) == 1
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            storage.get_trial(trial_ids[0])


def test_sqlite3_storage_rejects_apply_discard_without_migration() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/optuna.db"
        # A database created by Optuna has no `discarded_at` column.
        RDBStorage(f"sqlite:///{file_path}")

        with pytest.raises(RuntimeError, match="discarded_at"):
            rustuna.storages.SQLite3Storage(
                file_path, create_database=False, apply_discard=True
            )

        # create_database migrates the column in, so the same file is usable afterwards.
        storage = rustuna.storages.SQLite3Storage(
            file_path, create_database=True, apply_discard=True
        )
        assert storage.may_omit_trials()


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
