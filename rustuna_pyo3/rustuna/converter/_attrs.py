from __future__ import annotations

import json
from typing import Any

# Optuna exposes attrs as `dict[str, Any]`, while Rustuna stores them as `dict[str, str]`, so this
# converter reserves the below prefix for Optuna-facing attrs and stores their values as JSON.
# Prefixed keys are round-tripped through the compatibility layer, while non-prefixed keys remain
# Rustuna-internal attrs and are not exposed as Optuna attrs.
_OPTUNA_ATTR_PREFIX = "optuna_attr:"


def to_optuna_attrs(
    attrs: dict[str, str],
) -> dict[str, Any]:
    return {
        k[len(_OPTUNA_ATTR_PREFIX) :]: json.loads(v)
        for k, v in attrs.items()
        if k.startswith(_OPTUNA_ATTR_PREFIX)
    }


def to_rustuna_attrs(
    attrs: dict[str, Any],
) -> dict[str, str]:
    return {
        _OPTUNA_ATTR_PREFIX + k: json.dumps(v)
        for k, v in attrs.items()
        if not k.startswith(_OPTUNA_ATTR_PREFIX)
    }
