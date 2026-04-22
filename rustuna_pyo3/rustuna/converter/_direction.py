from __future__ import annotations

from collections.abc import Sequence

import optuna

import rustuna


def to_rustuna_directions(
    items: Sequence[optuna.study.StudyDirection],
) -> list[rustuna.study.StudyDirection]:
    return [to_rustuna_direction(item) for item in items]


def to_optuna_directions(
    items: Sequence[rustuna.study.StudyDirection],
) -> list[optuna.study.StudyDirection]:
    return [to_optuna_direction(item) for item in items]


def to_rustuna_direction(
    item: optuna.study.StudyDirection,
) -> rustuna.study.StudyDirection:
    if item == optuna.study.StudyDirection.MAXIMIZE:
        return rustuna.study.StudyDirection.MAXIMIZE
    elif item == optuna.study.StudyDirection.MINIMIZE:
        return rustuna.study.StudyDirection.MINIMIZE
    else:
        raise ValueError("Rustuna does not support StudyDirection.UNSET.")


def to_optuna_direction(
    item: rustuna.study.StudyDirection,
) -> optuna.study.StudyDirection:
    if item == rustuna.study.StudyDirection.MAXIMIZE:
        return optuna.study.StudyDirection.MAXIMIZE
    else:
        return optuna.study.StudyDirection.MINIMIZE
