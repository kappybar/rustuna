from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

import rustuna
from rustuna.converter import (
    to_rustuna_directions,
    to_rustuna_distribution,
    to_rustuna_distributions,
)
from rustuna.converter._storage import ToRustunaStorage
from rustuna.converter._trial import to_rustuna_state

if TYPE_CHECKING:
    from rustuna import SamplerProtocol, StorageProtocol


class ToOptunaSampler(BaseSampler):
    def __init__(self, sampler: SamplerProtocol) -> None:
        self._sampler = sampler
        self._inter_section_search_space = IntersectionSearchSpace()
        self._storage: rustuna.PyObjectStorage | None = None

    def _get_storage(self, storage: BaseStorage) -> rustuna.PyObjectStorage:
        if self._storage is None:
            self._storage = rustuna.PyObjectStorage(ToRustunaStorage(storage))
        return self._storage

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        ctx = rustuna.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        rustuna_search_space = to_rustuna_distributions(search_space)
        internal_params = self._sampler.sample_joint(ctx, storage, rustuna_search_space)
        external_params: dict[str, Any] = {}
        for param_name in internal_params:
            distribution = search_space[param_name]
            external_param_value = distribution.to_external_repr(
                internal_params[param_name]
            )
            external_params[param_name] = external_param_value
        return external_params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        ctx = rustuna.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        distribution = to_rustuna_distribution(param_distribution)
        internal_param = self._sampler.sample_independent(
            ctx, storage, param_name, distribution
        )
        return param_distribution.to_external_repr(internal_param)

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        if not self._sampler.support_joint_sampling:
            return {}

        # TODO(y0z): Support study.get_joint_search_space insead of using Optuna Python API
        # search_space = study.get_joint_search_space(study._study_id)
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._inter_section_search_space.calculate(
            study, use_cache=True
        ).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        after_trial = getattr(self._sampler, "after_trial", None)
        if after_trial is None:
            return

        ctx = rustuna.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        after_trial(
            ctx,
            storage,
            to_rustuna_state(state),
            list(values) if values is not None else None,
        )
