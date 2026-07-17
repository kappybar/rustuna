from __future__ import annotations

import tempfile

import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import rustuna


def test_reading_trials_after_late_user_attr_write_keeps_trials_readable() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.journal"
        storage = rustuna.storages.JournalFileStorage(file_path)
        study = rustuna.create_study(storage=storage, study_name="reproduce-journal")

        trial = study.ask()
        storage.set_trial_user_attrs(
            trial._trial_id,
            {
                "x": "1",
                "y": "2",
            },
        )
        study.tell(trial.number, values=1.0)

        with pytest.raises(rustuna.exceptions.UpdateFinishedTrialError):
            storage.set_trial_user_attrs(
                trial._trial_id,
                {
                    "x": "1",
                    "y": "2",
                },
            )

        trials = study.trials
        assert len(trials) == 1
        assert trials[0].number == trial.number


def test_create_new_trial_from_template_optuna_compatibility() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/rustuna.journal"
        rustuna_storage = rustuna.storages.JournalFileStorage(file_path)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage, study_name="example"
        )
        rustuna_study.add_trial(
            rustuna.trial.PersistedTrial(
                trial_id=0,
                study_id=0,
                number=0,
                state=rustuna.trial.TrialState.WAITING,
            )
        )

        optuna_storage = JournalStorage(JournalFileBackend(file_path))
        studies = optuna_storage.get_all_studies()
        assert len(studies) == 1
        trials = optuna_storage.get_all_trials(studies[0]._study_id, deepcopy=False)
        assert len(trials) == 1
        assert trials[0].state == optuna.trial.TrialState.WAITING


def test_journal_file_storage_can_load_discarded_trials() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/discarded.journal"
        storage = rustuna.storages.JournalFileStorage(file_path)
        study = rustuna.create_study(storage=storage, study_name="example")

        first = study.ask()
        second = study.ask()
        first_persisted = study.tell(first.number, 1.0)
        second_persisted = study.tell(second.number, 2.0)

        storage.discard_trials([first_persisted._trial_id])

        default_trials = storage.get_trials(study._study_id)
        assert [trial._trial_id for trial in default_trials] == [
            second_persisted._trial_id
        ]

        analysis_storage = rustuna.storages.JournalFileStorage(
            file_path,
            load_discarded_trials=True,
        )
        analysis_trials = analysis_storage.get_trials(study._study_id)
        assert [trial._trial_id for trial in analysis_trials] == [
            first_persisted._trial_id,
            second_persisted._trial_id,
        ]
        assert (
            analysis_storage.get_trial(first_persisted._trial_id)._trial_id
            == first_persisted._trial_id
        )
