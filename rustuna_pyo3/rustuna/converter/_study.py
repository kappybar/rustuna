from __future__ import annotations

import json

from optuna.study._frozen import FrozenStudy

import rustuna

from ._direction import to_optuna_directions, to_rustuna_directions


def to_frozen_study(study: rustuna.study.PersistedStudy) -> FrozenStudy:
    return FrozenStudy(
        study_name=study.name,
        study_id=study.id,
        direction=None,
        directions=to_optuna_directions(study.directions),
        user_attrs={k: json.loads(v) for k, v in study.user_attrs.items()},
        system_attrs={
            k: json.loads(v)
            for k, v in study.system_attrs.items()
            if not k.startswith("category_labels:")
        },
    )


def to_persisted_study(study: FrozenStudy) -> rustuna.study.PersistedStudy:
    return rustuna.study.PersistedStudy(
        id=study._study_id,
        name=study.study_name,
        directions=to_rustuna_directions(study.directions),
        # Prevent double encoding of JSON-serializable objects by checking if the value is already a string.
        # For example, "0" is a valid JSON-serializable value, but it should not be encoded again as '"0"'.
        # Note that this check is a heuristic and may not cover all cases.
        # If the value is a string that is not intended to be JSON-encoded, it will be stored as-is.
        user_attrs={
            k: v if isinstance(v, str) else json.dumps(v)
            for k, v in study.user_attrs.items()
        },
        system_attrs={
            k: v if isinstance(v, str) else json.dumps(v)
            for k, v in study.system_attrs.items()
        },
    )
