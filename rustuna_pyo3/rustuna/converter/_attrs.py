from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
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


class OptunaAttrsView(dict[str, Any]):
    def __init__(self, raw: Mapping[str, str]) -> None:
        super().__init__()
        self._raw = raw
        self._keys = tuple(
            key[len(_OPTUNA_ATTR_PREFIX) :]
            for key in raw.keys()
            if key.startswith(_OPTUNA_ATTR_PREFIX)
        )
        self._cache: dict[str, Any] = {}

    def _raw_key(self, key: str) -> str:
        return _OPTUNA_ATTR_PREFIX + key

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._keys:
            self._keys += (key,)
        self._cache[key] = value

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._keys

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        raw = self._raw[self._raw_key(key)]
        decoded = json.loads(raw)
        self._cache[key] = decoded
        return decoded

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):  # type: ignore[override]
        return ((key, self[key]) for key in self)

    def values(self):  # type: ignore[override]
        return (self[key] for key in self)

    def keys(self):  # type: ignore[override]
        return iter(self)

    def __copy__(self) -> dict[str, Any]:
        return dict(self.items())

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        copied = dict(self.items())
        memo[id(self)] = copied
        return copied

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            if len(self) != len(other):
                return False
            for key in self:
                if other.get(key) != self[key]:
                    return False
            return True
        return NotImplemented
