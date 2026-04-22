from __future__ import annotations

from typing import TYPE_CHECKING

from rustuna._rustuna import Distribution

if TYPE_CHECKING:
    from rustuna._rustuna import CategoricalChoiceType

__all__ = [
    "Distribution",
    "FloatDistribution",
    "IntDistribution",
    "CategoricalDistribution",
]


# TODO(c-bata): Replace FloatDistribution with a Python concrete class.
def FloatDistribution(
    low: float, high: float, log: bool = False, step: float | None = None
) -> Distribution:
    """Create a float distribution.

    Args:
        low: Lower bound.
        high: Upper bound.
        log: If True, sample from a log scale.
        step: Discretization step.

    Returns:
        A float distribution.
    """
    return Distribution.float(low=low, high=high, log=log, step=step)


# TODO(c-bata): Replace IntDistribution with a Python concrete class.
def IntDistribution(
    low: int, high: int, log: bool = False, step: int | None = None
) -> Distribution:
    """Create an integer distribution.

    Args:
        low: Lower bound.
        high: Upper bound.
        log: If True, sample from a log scale.
        step: Discretization step.

    Returns:
        An integer distribution.
    """
    if step is None:
        return Distribution.int(low=low, high=high, log=log)
    return Distribution.int(low=low, high=high, log=log, step=step)


# TODO(c-bata): Replace CategoricalDistribution with a Python concrete class.
def CategoricalDistribution(choices: list[CategoricalChoiceType]) -> Distribution:
    """Create a categorical distribution.

    Args:
        choices: List of candidate values.

    Returns:
        A categorical distribution.
    """
    return Distribution.categorical(choices=choices)
