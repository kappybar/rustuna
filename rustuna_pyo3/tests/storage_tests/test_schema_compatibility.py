from __future__ import annotations

import os.path
import tempfile

import optuna
import pytest
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

import rustuna
from rustuna.converter import ToOptunaStorage, ToRustunaStorage


def _objective(trial: optuna.Trial | rustuna.Trial) -> float:
    x = trial.suggest_float("x", 1.0, 10.0, log=True)
    trial.set_user_attr("json", '{"a": 1}')
    return x


def get_optuna_storage(backend: str, base_dir: str) -> optuna.storages.BaseStorage:
    if backend == "journal":
        file_path = os.path.join(base_dir, "test.journal")
        return JournalStorage(JournalFileBackend(file_path))
    if backend == "sqlite3":
        file_path = os.path.join(base_dir, "test.sqlite3")
        return RDBStorage(f"sqlite:///{file_path}")
    raise ValueError(f"Unknown backend: {backend}")


def get_rustuna_storage(
    backend: str, base_dir: str, create_database: bool
) -> rustuna.storages.OptunaStorageProtocol:
    if backend == "journal":
        file_path = os.path.join(base_dir, "test.journal")
        return rustuna.storages.JournalFileStorage(file_path)
    if backend == "sqlite3":
        file_path = os.path.join(base_dir, "test.sqlite3")
        return rustuna.storages.SQLite3Storage(
            file_path, create_database=create_database
        )
    raise ValueError(f"Unknown backend: {backend}")


@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
@pytest.mark.parametrize(
    "first_variant,second_variant",
    [
        ("direct", "via_to_optuna"),
        ("via_to_optuna", "direct"),
    ],
)
def test_optuna_api_resume_with_compat_storage(
    backend: str, first_variant: str, second_variant: str
) -> None:
    study_name = "compat-optuna"
    with tempfile.TemporaryDirectory() as workdir:
        # Start optimization via Optuna API
        if first_variant == "direct":
            first_storage = get_optuna_storage(backend, workdir)
        elif first_variant == "via_to_optuna":
            first_storage = ToOptunaStorage(
                get_rustuna_storage(backend, workdir, create_database=True)
            )
        else:
            raise ValueError(f"Unknown optuna variant: {first_variant}")

        first_study = optuna.create_study(storage=first_storage, study_name=study_name)
        first_study.optimize(_objective, n_trials=10)

        # Resume optimization via Optuna API
        if second_variant == "direct":
            second_storage = get_optuna_storage(backend, workdir)
        elif second_variant == "via_to_optuna":
            second_storage = ToOptunaStorage(
                get_rustuna_storage(backend, workdir, create_database=False)
            )
        else:
            raise ValueError(f"Unknown optuna variant: {second_variant}")

        second_study = optuna.load_study(storage=second_storage, study_name=study_name)
        second_study.optimize(_objective, n_trials=10)

        assert len(first_study.trials) == len(second_study.trials) == 20


@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
@pytest.mark.parametrize(
    "first_variant,second_variant",
    [
        ("direct", "via_to_rustuna"),
        ("via_to_rustuna", "direct"),
    ],
)
def test_rustuna_api_resume_with_compat_storage(
    backend: str, first_variant: str, second_variant: str
) -> None:
    study_name = "compat-rustuna"
    with tempfile.TemporaryDirectory() as workdir:
        # Start optimization via Rustuna API
        first_storage: rustuna.storages.StorageProtocol
        if first_variant == "direct":
            first_storage = get_rustuna_storage(backend, workdir, create_database=True)
        elif first_variant == "via_to_rustuna":
            first_storage = ToRustunaStorage(get_optuna_storage(backend, workdir))
        else:
            raise ValueError(f"Unknown rustuna variant: {first_variant}")

        first_study = rustuna.create_study(storage=first_storage, study_name=study_name)
        first_study.optimize(_objective, n_trials=10)

        # Resume optimization via Rustuna API
        second_storage: rustuna.storages.StorageProtocol
        if second_variant == "direct":
            second_storage = get_rustuna_storage(
                backend, workdir, create_database=False
            )
        elif second_variant == "via_to_rustuna":
            second_storage = ToRustunaStorage(get_optuna_storage(backend, workdir))
        else:
            raise ValueError(f"Unknown rustuna variant: {second_variant}")

        second_study = rustuna.load_study(storage=second_storage, study_name=study_name)
        second_study.optimize(_objective, n_trials=10)

        # TODO: Uncomment following line to align Rustuna Study to Optuna behavior.
        # assert len(first_study.trials) == len(second_study.trials) == 20
        assert (
            len(first_storage.get_trials(study_id=first_study._study_id))
            == len(second_storage.get_trials(study_id=first_study._study_id))
            == 20
        )


@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
def test_optuna_to_rustuna_resume(backend: str) -> None:
    study_name = "compat-optuna-to-rustuna"
    with tempfile.TemporaryDirectory() as workdir:
        optuna_storage = get_optuna_storage(backend, workdir)
        optuna_study = optuna.create_study(
            storage=optuna_storage, study_name=study_name
        )
        optuna_study.optimize(_objective, n_trials=10)

        rustuna_storage = get_rustuna_storage(backend, workdir, create_database=False)
        rustuna_study = rustuna.load_study(
            storage=rustuna_storage, study_name=study_name
        )
        rustuna_study.optimize(_objective, n_trials=10)

        assert len(optuna_study.trials) == len(rustuna_study.trials) == 20


@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
def test_rustuna_to_optuna_resume(backend: str) -> None:
    study_name = "compat-rustuna-to-optuna"
    with tempfile.TemporaryDirectory() as workdir:
        rustuna_storage = get_rustuna_storage(backend, workdir, create_database=True)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage, study_name=study_name
        )
        rustuna_study.optimize(_objective, n_trials=10)

        optuna_storage = get_optuna_storage(backend, workdir)
        optuna_study = optuna.load_study(storage=optuna_storage, study_name=study_name)
        optuna_study.optimize(_objective, n_trials=10)

        assert len(optuna_study.trials) == len(rustuna_study.trials) == 20
