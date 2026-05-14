from __future__ import annotations

from rustuna._protocols import SamplerProtocol
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
    """Create a random sampler.

    Args:
        seed: Random seed. If None, a random seed is used.

    Returns:
        A random sampler instance.
    """
    return Sampler.random(seed=seed)


# TODO(c-bata): Replace TPESampler with a Python concrete class.
def TPESampler(
    *,
    seed: int | None = None,
    n_startup_trials: int = 10,
    multivariate: bool = True,
) -> SamplerProtocol:
    """Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) `l(x)` to
    the set of parameter values associated with the good objective values, and another GMM
    `g(x)` to the remaining parameter values. It chooses the parameter value `x` that
    maximizes the ratio `l(x)/g(x)`. For multi-objective optimization, it uses non-domination
    ranks and hypervolume contributions to determine good and poor observations.

    For further information about the TPE algorithm, please refer to the following papers:

    - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
    - [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)
    - [Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance](https://arxiv.org/abs/2304.11127)

    For multi-objective TPE (MOTPE), please refer to the following papers:

    - [Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization Problems](https://doi.org/10.1145/3377930.3389817)
    - [Multiobjective Tree-Structured Parzen Estimator](https://doi.org/10.1613/jair.1.13188)

    Example:
        ```python
        import rustuna

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        sampler = rustuna.TPESampler(seed=42)
        study = rustuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=100)
        ```

    Args:
        seed: Seed for random number generator. If `None`, a random seed is used.
        n_startup_trials: The random sampling is used instead of the TPE algorithm until
            the given number of trials finish in the same study. Defaults to `10`.
        multivariate: If `True`, the multivariate TPE is used when suggesting parameters.
            The multivariate TPE samples all parameters jointly, which is reported to
            outperform the independent TPE. Defaults to `True`.

    Note:
        Multivariate mode is enabled by default (`multivariate=True`).
        In multivariate mode, TPE samples all non-conditional parameters jointly, which is reported to
        outperform independent sampling. See
        [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html)
        for more details.


    Returns:
        A TPE sampler instance.
    """
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
    """Sampler using NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm.

    NSGA-II is an evolutionary algorithm designed for multi-objective optimization. It maintains a
    population of candidates across generations, and uses non-dominated sorting to rank solutions
    and crowding distance to preserve diversity among Pareto-optimal solutions. Each generation,
    new candidates are generated via crossover and mutation of selected parents, and an elite
    selection strategy retains the best individuals for the next generation.

    For further information about the NSGA-II algorithm, please refer to the following paper:

    - [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://ieeexplore.ieee.org/document/996017)

    Example:
        ```python
        import rustuna

        def objective(trial):
            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_float("y", -5, 5)
            return x**2, y**2

        sampler = rustuna.NSGAIISampler(seed=42)
        study = rustuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=100)
        ```

    Args:
        seed: Seed for random number generator. If `None`, a random seed is used.
        population_size: Number of individuals in the population. Defaults to `50`.
        mutation_prob: Probability of mutating each parameter of a candidate. If `None`,
            `1.0 / len(search_space)` is used. Defaults to `None`.
        crossover_prob: Probability of performing crossover between two parents. Defaults to `0.9`.
        swapping_prob: Probability of swapping each parameter value during crossover.
            Defaults to `0.5`.

    Returns:
        An NSGA-II sampler instance.
    """
    return Sampler.nsgaii(
        seed=seed,
        population_size=population_size,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        swapping_prob=swapping_prob,
    )
