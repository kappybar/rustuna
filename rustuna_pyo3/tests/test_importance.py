from __future__ import annotations

from collections.abc import Callable

import pytest
from _pytest.fixtures import SubRequest
from optuna.importance import BaseImportanceEvaluator
from optuna.testing.pytest_importance import (
    BasicImportanceEvaluatorTestCase,
    ConditionalImportanceEvaluatorTestCase,
)

from rustuna.converter import ToOptunaImportanceEvaluator
from rustuna.importance import PedAnovaImportanceEvaluator


class TestBasicImportanceEvaluator(BasicImportanceEvaluatorTestCase):
    @pytest.fixture(params=[PedAnovaImportanceEvaluator])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return lambda: ToOptunaImportanceEvaluator(request.param())

    # TODO(kAIto47802): Remove these skip once Rustuna's PED-ANOVA supports target.
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
    @pytest.mark.parametrize(
        "target_idx",
        [
            pytest.param(
                0,
                marks=pytest.mark.skip(
                    reason="Rustuna does not support target yet"
                ),
            ),
            pytest.param(
                1,
                marks=pytest.mark.skip(
                    reason="Rustuna does not support target yet"
                ),
            ),
            None,
        ],
    )
    def test_evaluator_with_infinite(
        self,
        evaluator: Callable[..., BaseImportanceEvaluator],
        inf_value: float,
        target_idx: int | None,
    ) -> None:
        super().test_evaluator_with_infinite(
            evaluator,
            inf_value,
            target_idx,
        )

    # TODO(kAIto47802): Remove this skip once Rustuna's PED-ANOVA supports target.
    @pytest.mark.skip(reason="Rustuna does not support target yet")
    def test_importance_evaluator_with_target(
        self,
        evaluator: Callable[..., BaseImportanceEvaluator],
    ) -> None:
        super().test_importance_evaluator_with_target(evaluator)


class TestConditionalImportanceEvaluator(ConditionalImportanceEvaluatorTestCase):
    @pytest.fixture(params=[PedAnovaImportanceEvaluator])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return lambda: ToOptunaImportanceEvaluator(request.param())
