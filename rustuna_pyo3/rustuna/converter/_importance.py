from __future__ import annotations

from collections.abc import Callable

import optuna
from optuna.importance import BaseImportanceEvaluator
from optuna.trial import FrozenTrial

import rustuna
from rustuna.converter._storage import ToRustunaStorage


class ToOptunaImportanceEvaluator(BaseImportanceEvaluator):
    def __init__(
        self,
        evaluator: rustuna.importance.PedAnovaImportanceEvaluator,
    ) -> None:
        self._evaluator = evaluator

    # TODO(kAIto47802): Support the `target` argument in Rustuna's PED-ANOVA evaluator.
    def evaluate(
        self,
        study: optuna.Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        rustuna_study = rustuna.load_study(
            study_name=study.study_name, storage=ToRustunaStorage(study._storage)
        )
        return self._evaluator.evaluate(rustuna_study, params=params)
