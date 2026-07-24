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

    def evaluate(
        self,
        study: optuna.Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        rustuna_study = rustuna.load_study(
            study.study_name, ToRustunaStorage(study._storage)
        )

        return self._evaluator.evaluate(rustuna_study, params=params)
