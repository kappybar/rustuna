from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import numpy as np
from optuna._transform import _SearchSpaceTransform
from optuna.importance import BaseImportanceEvaluator
from optuna.trial import TrialState

from rustuna._rustuna import _get_param_importance_from_list

if TYPE_CHECKING:
    from typing import Callable, Iterable

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    def __init__(
        self,
        *,
        n_trees: int = 64,
        max_depth: int = 64,
        seed: int | None = None,
        completed_trials: list[FrozenTrial] | None = None,
    ) -> None:
        # TODO(c-bata): Support constructor options.
        self._completed_trials = completed_trials
        self._n_trees = n_trees

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        if self._completed_trials is None:
            completed_trials = study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,)
            )
        else:
            completed_trials = self._completed_trials

        _fast_check_evaluate_args(completed_trials, params)
        if params is None:
            distributions = _fast_intersection_search_space(completed_trials)
        else:
            distributions = _fast_get_distributions(completed_trials, params)

        if len(distributions) == 0:
            return {}

        # fANOVA does not support parameter distributions with a single value.
        # However, there is no reason to calculate parameter importance in such case anyway,
        # since it will always be 0 as the parameter is constant in the objective function.
        zero_importances = {
            name: 0.0 for name, dist in distributions.items() if dist.single()
        }
        distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }

        param_names = list(distributions.keys())
        trials: list[FrozenTrial] = []
        features: list[list[float]] = [[] for _ in param_names]
        targets: list[float] = []
        for trial in _filter_nonfinite(completed_trials, target=target):
            if any(name not in trial.params for name in distributions.keys()):
                continue
            for i, name in enumerate(param_names):
                distribution = distributions[name]
                features[i].append(distribution.to_internal_repr(trial.params[name]))
            target_value = trial.value if target is None else target(trial)
            targets.append(cast(float, target_value))

        importances = _get_param_importance_from_list(features, targets, self._n_trees)
        return {param_name: importances[i] for i, param_name in enumerate(param_names)}


def _fast_intersection_search_space(
    completed_trials: list[FrozenTrial],
) -> dict[str, BaseDistribution]:
    search_space = None
    for trial in reversed(completed_trials):
        if search_space is None:
            search_space = trial.distributions
            continue

        delete_list = []
        for param_name, param_distribution in search_space.items():
            if param_name not in trial.distributions:
                delete_list.append(param_name)
            elif trial.distributions[param_name] != param_distribution:
                delete_list.append(param_name)

        for param_name in delete_list:
            del search_space[param_name]

    search_space = search_space or {}
    return {k: copy.copy(search_space[k]) for k in sorted(search_space)}


def _fast_check_evaluate_args(
    completed_trials: list[FrozenTrial], params: list[str] | None
) -> None:
    if len(completed_trials) == 0:
        raise ValueError(
            "Cannot evaluate parameter importances without completed trials."
        )
    if len(completed_trials) == 1:
        raise ValueError(
            "Cannot evaluate parameter importances with only a single trial."
        )

    if params is not None:
        if not isinstance(params, (list, tuple)):
            raise TypeError(
                "Parameters must be specified as a list. Actual parameters: {}.".format(
                    params
                )
            )
        if any(not isinstance(p, str) for p in params):
            raise TypeError(
                "Parameters must be specified by their names with strings. Actual parameters: "
                "{}.".format(params)
            )

        if len(params) > 0:
            at_least_one_trial = False
            for trial in completed_trials:
                if all(p in trial.distributions for p in params):
                    at_least_one_trial = True
                    break
            if not at_least_one_trial:
                raise ValueError(
                    "Study must contain completed trials with all specified parameters. "
                    "Specified parameters: {}.".format(params)
                )


def _fast_get_distributions(
    completed_trials: list[FrozenTrial], params: list[str] | None
) -> dict[str, BaseDistribution]:
    # New temporary required to pass mypy. Seems like a bug.
    params_not_none = params
    assert params_not_none is not None

    # Compute the search space based on the subset of trials containing all parameters.
    distributions = None
    for trial in completed_trials:
        trial_distributions = trial.distributions
        if not all(name in trial_distributions for name in params_not_none):
            continue

        if distributions is None:
            distributions = {
                k: trial_distributions[k]
                for k in trial_distributions
                if k in params_not_none
            }
            continue

        if any(
            trial_distributions[name] != distribution
            for name, distribution in distributions.items()
        ):
            raise ValueError(
                "Parameters importances cannot be assessed with dynamic search spaces if "
                "parameters are specified. Specified parameters: {}.".format(params)
            )

    assert distributions is not None  # Required to pass mypy.
    return {k: copy.copy(distributions[k]) for k in sorted(distributions)}


def _filter_nonfinite(
    trials: Iterable[FrozenTrial],
    target: Callable[[FrozenTrial], float] | None = None,
) -> list[FrozenTrial]:
    # For multi-objective optimization target must be specified to select
    # one of objective values to filter trials by (and plot by later on).
    # This function is not raising when target is missing, sice we're
    # assuming plot args have been sanitized before.
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    filtered_trials: list[FrozenTrial] = []
    for trial in trials:
        # Not a Number, positive infinity and negative infinity are considered to be non-finite.
        if np.isfinite(target(trial)):
            filtered_trials.append(trial)
    return filtered_trials
