from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rustuna._rustuna import SamplerProtocol

from rustuna._rustuna import Sampler, SamplerContext

__all__ = [
    "SamplerContext",
    "SamplerProtocol",
    "RandomSampler",
    "TPESampler",
    "NSGAIISampler",
]


# TODO(c-bata): Replace RandomSampler with a Python concrete class.
def RandomSampler(*, seed: int | None = None) -> SamplerProtocol:
    return Sampler.random(seed=seed)


# TODO(c-bata): Replace TPESampler with a Python concrete class.
def TPESampler(
    *,
    seed: int | None = None,
    n_startup_trials: int = 10,
    multivariate: bool = True,
) -> SamplerProtocol:
    return Sampler.tpe(
        seed=seed, n_startup_trials=n_startup_trials, multivariate=multivariate
    )


# TODO(c-bata): Replace NSGAIISampler with a Python concrete class.
def NSGAIISampler(
    *,
    seed: int | None = None,
    population_size: int = 50,
    mutation_prob: float | None = None,
    crossover_prob: float = 0.9,
    swapping_prob: float = 0.5,
) -> SamplerProtocol:
    return Sampler.nsgaii(
        seed=seed,
        population_size=population_size,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        swapping_prob=swapping_prob,
    )
