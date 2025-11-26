from __future__ import annotations

import typing

import optuna

import rustuna

from ._distribution import to_optuna_distribution, to_rustuna_distribution
from ._study import to_optuna_directions, to_persisted_study
from ._trial import to_frozen_trial, to_optuna_state, to_persisted_trial

if typing.TYPE_CHECKING:
    from optuna._typing import JSONSerializable
    from optuna.storages import BaseStorage


class ToRustunaStorage:
    def __init__(self, storage: BaseStorage, is_distributed: bool = False) -> None:
        self._storage = storage
        self._is_distributed = is_distributed

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    def create_new_study(
        self, study_name: str, directions: list[rustuna.StudyDirection]
    ) -> rustuna.PersistedStudy:
        optuna_directions = to_optuna_directions(directions)
        study_id = self._storage.create_new_study(optuna_directions, study_name)
        return rustuna.PersistedStudy(
            id=study_id,
            name=study_name,
            directions=directions,
            user_attrs={},
            system_attrs={},
        )

    def create_new_trial(self, study_id: int) -> rustuna.PersistedTrial:
        trial_id = self._storage.create_new_trial(study_id)
        trial = self._storage.get_trial(trial_id)
        return to_persisted_trial(trial, study_id)

    def set_trial_param(
        self,
        study_id: int,
        trial_number: int,
        name: str,
        distribution: rustuna.Distribution,
        value: float,
    ) -> None:
        trial_id = self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )
        self._storage.set_trial_param(
            trial_id, name, value, to_optuna_distribution(distribution)
        )

    def set_trial_state_values(
        self,
        study_id: int,
        trial_number: int,
        state: rustuna.TrialState,
        values: None | list[float] = None,
    ) -> None:
        trial_id = self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )
        self._storage.set_trial_state_values(
            trial_id,
            to_optuna_state(state),
            values=values,
        )

    def get_studies(self) -> list[rustuna.PersistedStudy]:
        frozen_studies = self._storage.get_all_studies()
        return [to_persisted_study(s) for s in frozen_studies]

    def get_study(self, study_id: int) -> rustuna.PersistedStudy:
        frozen_studies = self._storage.get_all_studies()
        for s in frozen_studies:
            if s._study_id == study_id:
                return to_persisted_study(s)
        raise KeyError(f"Study {study_id} not found")

    def get_trials(self, study_id: int) -> list[rustuna.PersistedTrial]:
        frozen_trials = self._storage.get_all_trials(study_id)
        return [to_persisted_trial(t, study_id) for t in frozen_trials]

    def get_trial(self, study_id: int, trial_number: int) -> rustuna.PersistedTrial:
        trial_id = self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )
        frozen_trial = self._storage.get_trial(trial_id)
        return to_persisted_trial(frozen_trial, study_id)

    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_study_system_attr(study_id, key, value)

    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_study_user_attr(study_id, key, value)

    def set_trial_system_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None:
        trial_id = self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )
        for key, value in attrs.items():
            self._storage.set_trial_system_attr(trial_id, key, value)

    def set_trial_user_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None:
        trial_id = self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )
        for key, value in attrs.items():
            self._storage.set_trial_user_attr(trial_id, key, value)
