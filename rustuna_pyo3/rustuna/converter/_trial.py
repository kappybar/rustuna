from __future__ import annotations

import datetime
import json
import typing

import optuna
from optuna.trial import FrozenTrial, TrialState

import rustuna

from ._attr import to_optuna_attrs, to_rustuna_attrs
from ._distribution import (
    to_optuna_distributions,
    to_rustuna_distribution,
    to_rustuna_distributions,
)

# This is a dummy datetime since Rustuna does not store the datetime_start and datetime_complete.
dummy_datetime = datetime.datetime(
    2023, 11, 26, 16, 56, 38
)  # Date of the initial commit of Rustuna

to_rustuna_state_map = {
    optuna.trial.TrialState.RUNNING: rustuna.TrialState.RUNNING,
    optuna.trial.TrialState.COMPLETE: rustuna.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL: rustuna.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED: rustuna.TrialState.PRUNED,
    optuna.trial.TrialState.WAITING: rustuna.TrialState.WAITING,
}
# TODO(c-bata): Make rustuna.TrialState hashable.
# to_optuna_state_map = {v: k for k, v in to_optuna_state_map.items()}


def to_optuna_state(state: rustuna.TrialState) -> optuna.trial.TrialState:
    if state == rustuna.TrialState.RUNNING:
        return optuna.trial.TrialState.RUNNING
    elif state == rustuna.TrialState.COMPLETE:
        return optuna.trial.TrialState.COMPLETE
    elif state == rustuna.TrialState.FAIL:
        return optuna.trial.TrialState.FAIL
    elif state == rustuna.TrialState.PRUNED:
        return optuna.trial.TrialState.PRUNED
    elif state == rustuna.TrialState.WAITING:
        return optuna.trial.TrialState.WAITING
    else:
        raise KeyError(f"Unknown state: {state}")


def to_rustuna_state(state: optuna.trial.TrialState) -> rustuna.TrialState:
    return to_rustuna_state_map[state]


def to_persisted_trial(
    trial: optuna.trial.FrozenTrial,
    study_id: int,
) -> rustuna.PersistedTrial:
    optuna_system_attrs = trial.system_attrs.copy()
    if trial.datetime_start is not None:
        optuna_system_attrs["datetime_start"] = trial.datetime_start.isoformat(
            timespec="microseconds"
        )
    if trial.datetime_complete is not None:
        optuna_system_attrs["datetime_complete"] = trial.datetime_complete.isoformat(
            timespec="microseconds"
        )
    if trial.intermediate_values:
        optuna_system_attrs["intermediate_values"] = trial.intermediate_values

    internal_params: dict[str, float] = {}
    distributions: dict[str, rustuna.Distribution] = {}
    for param_name in trial.distributions:
        optuna_distribution = trial.distributions[param_name]
        distributions[param_name] = to_rustuna_distribution(optuna_distribution)
        internal_params[param_name] = optuna_distribution.to_internal_repr(
            trial.params[param_name]
        )

    return rustuna.PersistedTrial(
        study_id=study_id,
        number=trial.number,
        state=to_rustuna_state(trial.state),
        values=trial.values,
        internal_params=internal_params,
        distributions=distributions,
        user_attrs=to_rustuna_attrs(trial.user_attrs),
        system_attrs=to_rustuna_attrs(optuna_system_attrs),
    )


def to_frozen_trial(
    persisted_study: rustuna.PersistedStudy,
    persisted_trial: rustuna.PersistedTrial,
    trial_id: int,
) -> FrozenTrial:
    optuna_system_attrs = to_optuna_attrs(persisted_trial.system_attrs)
    if "datetime_start" in optuna_system_attrs:
        datetime_start = datetime.datetime.fromisoformat(
            typing.cast(str, optuna_system_attrs["datetime_start"])
        )
    elif persisted_trial.state != rustuna.TrialState.WAITING:
        datetime_start = dummy_datetime
    else:
        datetime_start = None

    if "datetime_complete" in optuna_system_attrs:
        datetime_complete = datetime.datetime.fromisoformat(
            typing.cast(str, optuna_system_attrs["datetime_complete"])
        )
    elif persisted_trial.state.is_finished():
        # Add 1 second to pass the Optuna's trial validation.
        datetime_complete = dummy_datetime + datetime.timedelta(seconds=1)
    else:
        datetime_complete = None

    intermediate_values = {}
    if optuna_system_attrs.get("intermediate_values"):
        intermediate_values = json.loads(
            typing.cast(str, optuna_system_attrs["intermediate_values"])
        )
        intermediate_values = {
            int(step): value for step, value in intermediate_values.items()
        }
    return FrozenTrial(
        trial_id=trial_id,
        number=persisted_trial.number,
        value=None,
        state=to_optuna_state(persisted_trial.state),
        values=persisted_trial.values,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=persisted_trial.params,
        distributions=to_optuna_distributions(persisted_trial.distributions),
        user_attrs=to_optuna_attrs(persisted_trial.user_attrs),
        system_attrs=optuna_system_attrs,
        intermediate_values=intermediate_values,
    )
