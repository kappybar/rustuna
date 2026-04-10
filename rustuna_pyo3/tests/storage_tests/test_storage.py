from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest
from optuna.storages import InMemoryStorage, JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from pytest import FixtureRequest

import rustuna
from rustuna.converter._storage import ToRustunaStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from rustuna import StorageProtocol


@pytest.fixture(
    params=[
        "inmemory",
        "sqlite3",
        "journal-file",
        # TODO(c-bata): Fix test_studyw_get_user_attr_method
        # "optuna-inmemory",
        # "optuna-rdb-sqlite3",
        # "optuna-journal-file",
    ]
)
def storage(request: FixtureRequest) -> Generator[StorageProtocol, None, None]:
    if request.param == "inmemory":
        yield rustuna.Storage.in_memory()
        return
    if request.param == "optuna-inmemory":
        yield ToRustunaStorage(InMemoryStorage())
        return
    if request.param == "optuna-rdb-sqlite3":
        yield ToRustunaStorage(RDBStorage("sqlite://"))
        return

    with tempfile.TemporaryDirectory() as workdir:
        if request.param == "sqlite3":
            file_path = f"{workdir}/test.db"
            yield rustuna.Storage.sqlite3(file_path, create_database=True)
        elif request.param == "optuna-journal-file":
            file_path = f"{workdir}/test.journal"
            yield ToRustunaStorage(JournalStorage(JournalFileBackend(file_path)))
        else:
            file_path = f"{workdir}/test.journal"
            yield rustuna.Storage.journal_file(file_path)


def test_get_study_attr_methods(storage: StorageProtocol) -> None:
    study = storage.create_new_study("example study", [rustuna.StudyDirection.MINIMIZE])
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "study_user_attr",
        },
    )
    storage.set_study_system_attrs(
        study.id,
        {
            "study_system_attr": "study_system_attr",
        },
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "study_user_attr"
    assert (
        storage.get_study_system_attr(study.id, "study_system_attr")
        == "study_system_attr"
    )
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "updated",
        },
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "updated"


def test_study_get_user_attr_method(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)
    study.set_user_attr("key", "1")
    assert study.get_user_attr("key", decoder=int) == 1
    assert study.get_user_attr("not_found") is None
    assert study.get_user_attr("not_found", default=1) == 1


def test_get_trial_attr_methods(storage: StorageProtocol) -> None:
    study = storage.create_new_study("example study", [rustuna.StudyDirection.MINIMIZE])
    trial = storage.create_new_trial(study.id)
    storage.set_trial_user_attrs(
        trial._trial_id,
        {
            "trial_user_attr": "trial_user_attr",
        },
    )
    storage.set_trial_system_attrs(
        trial._trial_id,
        {
            "trial_system_attr": "trial_system_attr",
        },
    )
    assert (
        storage.get_trial_user_attr(trial._trial_id, "trial_user_attr")
        == "trial_user_attr"
    )
    assert (
        storage.get_trial_system_attr(trial._trial_id, "trial_system_attr")
        == "trial_system_attr"
    )
    storage.set_trial_user_attrs(
        trial._trial_id,
        {
            "trial_user_attr": "updated",
        },
    )
    assert storage.get_trial_user_attr(trial._trial_id, "trial_user_attr") == "updated"


def test_trial_get_user_attr_method(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)
    trial = study.ask()
    trial.set_user_attr("key", "1")
    persisted = study.tell(trial.number, 0.0)
    assert persisted.get_user_attr("key", decoder=int) == 1
    assert persisted.get_user_attr("not_found") is None
    assert persisted.get_user_attr("not_found", default=1) == 1
