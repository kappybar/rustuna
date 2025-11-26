from __future__ import annotations

import pytest

import rustuna


class DummyIndependentSampler:
    def __init__(self) -> None:
        pass

    @property
    def support_joint_sampling(self) -> bool:
        return False

    def sample_joint(
        self,
        ctx: rustuna.SamplerContext,
        storage: rustuna.Storage,
        search_space: dict[str, rustuna.Distribution],
    ) -> dict[str, float]:
        assert False, "Unreachable code"

    def sample_independent(
        self,
        ctx: rustuna.SamplerContext,
        storage: rustuna.Storage,
        name: str,
        distribution: rustuna.Distribution,
    ) -> float:
        dic = distribution.to_dict()
        if dic["type"] == "FloatDistribution":
            return dic["low"]
        elif dic["type"] == "IntDistribution":
            return dic["low"]
        elif dic["type"] == "CategoricalDistribution":
            return 0
        assert False, "Unreachable code"


class DummyJointSampler:
    def __init__(self) -> None:
        pass

    @property
    def support_joint_sampling(self) -> bool:
        return True

    def sample_joint(
        self,
        ctx: rustuna.SamplerContext,
        storage: rustuna.Storage,
        search_space: dict[str, rustuna.Distribution],
    ) -> dict[str, float]:
        params = {}
        for name, distribution in search_space.items():
            dic = distribution.to_dict()
            if dic["type"] == "FloatDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "IntDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "CategoricalDistribution":
                params[name] = 0
            else:
                assert False, "Unreachable code"
        return params

    def sample_independent(
        self,
        ctx: rustuna.SamplerContext,
        storage: rustuna.Storage,
        name: str,
        distribution: rustuna.Distribution,
    ) -> float:
        dic = distribution.to_dict()
        if dic["type"] == "FloatDistribution":
            return dic["low"]
        elif dic["type"] == "IntDistribution":
            return dic["low"]
        elif dic["type"] == "CategoricalDistribution":
            return 0
        assert False, "Unreachable code"


@pytest.mark.parametrize("sampler", [DummyIndependentSampler(), DummyJointSampler()])
def test_custom_sampler(sampler: rustuna.SamplerProtocol) -> None:
    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        value = (x - 2) ** 2 + (y + 5) ** 2
        return value

    study = rustuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
