from __future__ import annotations

import json
import typing

import optuna

import rustuna

if typing.TYPE_CHECKING:
    from optuna._typing import JSONSerializable


def to_optuna_attrs(
    src: dict[str, str],
) -> dict[str, JSONSerializable]:
    return {k: json.loads(v) for k, v in src.items()}


def to_rustuna_attrs(
    src: dict[str, JSONSerializable],
) -> dict[str, str]:
    return {k: json.dumps(v) for k, v in src.items()}
