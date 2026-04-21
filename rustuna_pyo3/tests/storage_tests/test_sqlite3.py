import tempfile

import optuna
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
