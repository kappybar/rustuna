from __future__ import annotations

from collections.abc import Container
from typing import Any

import optuna
from optuna import Study as _OptunaStudy

import rustuna

from ._storage import ToOptunaStorage
from ._trial import (
    to_frozen_trial,
    to_rustuna_state_map,
)


class ToOptunaStudy(_OptunaStudy):
    def __init__(self, study: rustuna.Study):
        self._rustuna_study = study
        super().__init__(
            study.study_name,
            ToOptunaStorage(study._storage),
        )

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Container[optuna.trial.TrialState] | None = None,
    ) -> list[optuna.trial.FrozenTrial]:
        rustuna_states: list[rustuna.trial.TrialState] | None = None
        if states is not None:
            rustuna_states = []
            for optuna_state, rustuna_state in to_rustuna_state_map.items():
                if optuna_state not in states:
                    continue
                rustuna_states.append(rustuna_state)
        trials = self._rustuna_study.get_trials(states=rustuna_states)
        return [to_frozen_trial(t, use_frozen_trial_like=True) for t in trials]

    @property
    def user_attrs(self) -> dict[str, Any]:
        return self._rustuna_study.user_attrs
