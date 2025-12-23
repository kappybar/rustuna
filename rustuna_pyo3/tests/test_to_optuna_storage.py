from __future__ import annotations

import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState

import rustuna
from rustuna.converter import ToOptunaStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from optuna.storages import BaseStorage


@pytest.fixture
def storage() -> Generator[BaseStorage, None, None]:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"
        rustuna_storage = rustuna.Storage.sqlite3(file_path, create_database=True)
        storage = ToOptunaStorage(rustuna_storage)

        yield storage


class TestSQLite3Storage(StorageTestCase):
    def test_delete_study(self, storage: BaseStorage) -> None:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.create_new_trial(study_id)
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1

        # TODO(c-bata): Check study_id
        # with pytest.raises(KeyError):
        #     # Deletion of non-existent study.
        #     storage.delete_study(study_id + 1)

        storage.delete_study(study_id)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 0

        # storage.delete_study(study_id)
        # with pytest.raises(KeyError):
        #     # Double free.
        #     storage.delete_study(study_id)

    @pytest.mark.parametrize(
        "start_time,complete_time",
        [
            (datetime.now(), datetime.now()),
            (datetime(2022, 9, 1), datetime(2022, 9, 2)),
        ],
    )
    def test_create_new_trial_with_template_trial(
        self, storage: BaseStorage, start_time: datetime, complete_time: datetime
    ) -> None:
        template_trial = FrozenTrial(
            state=TrialState.COMPLETE,
            value=10000,
            datetime_start=start_time,
            datetime_complete=complete_time,
            params={"x": 0.5},
            distributions={"x": FloatDistribution(0, 1)},
            user_attrs={"foo": "bar"},
            system_attrs={"baz": 123},
            intermediate_values={1: 10, 2: 100, 3: 1000},
            number=55,  # This entry is ignored.
            trial_id=-1,  # dummy value (unused).
        )

        def _check_trials(trials: list[FrozenTrial], idx: int, trial_id: int) -> None:
            assert len(trials) == idx + 1
            assert len({t._trial_id for t in trials}) == idx + 1
            assert trial_id in {t._trial_id for t in trials}
            assert {t.number for t in trials} == set(range(idx + 1))
            assert all(t.state == template_trial.state for t in trials)
            assert all(t.params == template_trial.params for t in trials)
            assert all(t.distributions == template_trial.distributions for t in trials)
            # TODO(c-bata): Support intermediate_values
            # assert all(t.intermediate_values == template_trial.intermediate_values for t in trials)
            assert all(t.user_attrs == template_trial.user_attrs for t in trials)
            assert all(t.system_attrs == template_trial.system_attrs for t in trials)
            # TODO(c-bata): Support to copy datetime_start and datetime_complete
            # assert all(t.datetime_start == template_trial.datetime_start for t in trials)
            # assert all(t.datetime_complete == template_trial.datetime_complete for t in trials)
            assert all(t.value == template_trial.value for t in trials)

        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            trial_id = storage.create_new_trial(study_id, template_trial=template_trial)
            trials = storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id)

        # Create trial in non-existent study.
        with pytest.raises(KeyError):
            storage.create_new_trial(study_id + 1)

        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        for i in range(n_trial_in_study):
            storage.create_new_trial(study_id2, template_trial=template_trial)
            trials = storage.get_all_trials(study_id2)
            assert {t.number for t in trials} == set(range(i + 1))

        trials = storage.get_all_trials(study_id) + storage.get_all_trials(study_id2)
        # Check trial_ids are unique across studies.
        assert len({t._trial_id for t in trials}) == 2 * n_trial_in_study

    @pytest.mark.skip("Fix me!")
    def test_set_trial_state_values_for_values(self, storage: BaseStorage) -> None:
        super().test_set_trial_state_values_for_values(storage)

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_trial(self, storage: BaseStorage) -> None:
        super().test_get_trial(storage)

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_all_trials(self, storage: BaseStorage) -> None:
        super().test_get_all_trials(storage)

    @pytest.mark.skip("Rustuna's params cannot support the order of params")
    @pytest.mark.parametrize("param_names", [["a", "b"], ["b", "a"]])
    def test_get_all_trials_params_order(
        self, storage: BaseStorage, param_names: list[str]
    ) -> None: ...

    @pytest.mark.skip("Rustuna's SQLite3 storage does not support pickle serialization")
    def test_pickle_storage(self, storage: BaseStorage) -> None: ...
