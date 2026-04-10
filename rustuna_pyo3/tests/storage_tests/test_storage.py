from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING

import pytest
from pytest import FixtureRequest

import rustuna


if TYPE_CHECKING:
    from collections.abc import Generator
    from rustuna import StorageProtocol


@pytest.fixture(params=["in_memory", "sqlite3", "journal-file"])
def storage(request: FixtureRequest) -> Generator[StorageProtocol, None, None]:
    if request.param == "in_memory":
        yield rustuna.Storage.in_memory()

    with tempfile.TemporaryDirectory() as workdir:
        if request.param == "sqlite3":
            file_path = f"{workdir}/test.db"
            yield rustuna.Storage.sqlite3(file_path, create_database=True)
        else:
            file_path = f"{workdir}/test.journal"
            yield rustuna.Storage.journal_file(file_path)


def test_get_study_attr_methods(storage: StorageProtocol) -> None:
    study = storage.create_new_study("example study", [rustuna.StudyDirection.MINIMIZE])
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "study_user_attr",
        }
    )
    storage.set_study_system_attrs(
        study.id,
        {
            "study_system_attr": "study_system_attr",
        }
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "study_user_attr"
    assert storage.get_study_system_attr(study.id, "study_system_attr") == "study_system_attr"
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "updated",
        }
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "updated"



def test_get_trial_attr_methods(storage: StorageProtocol) -> None:
    study = storage.create_new_study("example study", [rustuna.StudyDirection.MINIMIZE])
    trial = storage.create_new_trial(study.id)
    storage.set_trial_user_attrs(
        trial._trial_id,
        {
            "trial_user_attr": "trial_user_attr",
        }
    )
    storage.set_trial_system_attrs(
        trial._trial_id,
        {
            "trial_system_attr": "trial_system_attr",
        }
    )
    assert storage.get_trial_user_attr(trial._trial_id, "trial_user_attr") == "trial_user_attr"
    assert storage.get_trial_system_attr(trial._trial_id, "trial_system_attr") == "trial_system_attr"
    storage.set_trial_user_attrs(
        trial._trial_id,
        {
            "trial_user_attr": "updated",
        }
    )
    assert storage.get_trial_user_attr(trial._trial_id, "trial_user_attr") == "updated"
