from __future__ import annotations

import optuna

import rustuna
from rustuna._rustuna import Distribution


def to_optuna_distributions(
    src: dict[str, Distribution],
) -> dict[str, optuna.distributions.BaseDistribution]:
    return {k: to_optuna_distribution(v) for k, v in src.items()}


def to_rustuna_distributions(
    src: dict[str, optuna.distributions.BaseDistribution],
) -> dict[str, Distribution]:
    return {k: to_rustuna_distribution(v) for k, v in src.items()}


def to_optuna_distribution(
    src: Distribution,
) -> optuna.distributions.BaseDistribution:
    d = src.to_dict()
    if d["type"] == "FloatDistribution":
        return optuna.distributions.FloatDistribution(
            low=d["low"], high=d["high"], log=d["log"], step=d["step"]
        )
    elif d["type"] == "IntDistribution":
        return optuna.distributions.IntDistribution(
            low=d["low"], high=d["high"], log=d["log"], step=d["step"] or 1
        )
    elif d["type"] == "CategoricalDistribution":
        return optuna.distributions.CategoricalDistribution(choices=d["choices"])
    else:
        raise ValueError(f"Unknown distribution type: {d['type']}")


def to_rustuna_distribution(
    src: optuna.distributions.BaseDistribution,
) -> Distribution:
    if isinstance(src, optuna.distributions.FloatDistribution):
        return rustuna.distributions.FloatDistribution(
            low=src.low,
            high=src.high,
            log=src.log,
            step=src.step,
        )
    elif isinstance(src, optuna.distributions.IntDistribution):
        return rustuna.distributions.IntDistribution(
            low=src.low,
            high=src.high,
            log=src.log,
            step=src.step,
        )
    elif isinstance(src, optuna.distributions.CategoricalDistribution):
        return rustuna.distributions.CategoricalDistribution(choices=list(src.choices))
    else:
        raise ValueError(f"Unknown distribution: {src}")
