from __future__ import annotations

import json
from typing import Any

# Optuna exposes attrs as `dict[str, Any]`, while Rustuna stores them as
# `dict[str, str]`. String values are stored under their original keys so that
# Rustuna-native and Optuna-originated string attrs have the same representation.
# Non-string Optuna values are JSON-encoded under their original keys and are
# accompanied by a marker key. The marker lets `to_optuna_attrs` restore the
# original type without attempting to JSON-decode ordinary Rustuna strings.
#
# The marker key is an implementation detail and is omitted when converting
# attrs back to Optuna. Rustuna may also store internal system attrs, such as
# categorical labels; callers exposing attrs through Optuna are responsible for
# filtering those Rustuna-specific attributes when necessary.
_PREFIX_REQUIRE_JSON_ENCODE = "optuna_json_encoded:"


def to_optuna_attrs(attrs: dict[str, str]) -> dict[str, Any]:
    converted = {}
    for key, value in attrs.items():
        if key.startswith(_PREFIX_REQUIRE_JSON_ENCODE):
            continue
        if _PREFIX_REQUIRE_JSON_ENCODE + key in attrs:
            converted[key] = json.loads(value)
        else:
            converted[key] = value
    return converted


def to_rustuna_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    converted = {}
    for key, value in attrs.items():
        if isinstance(value, str):
            converted[key] = value
            continue
        converted[key] = json.dumps(value)
        converted[_PREFIX_REQUIRE_JSON_ENCODE + key] = "true"
    return converted
