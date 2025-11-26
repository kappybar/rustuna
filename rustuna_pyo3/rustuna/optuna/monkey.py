def patch_all() -> None:
    patch_importance()


def patch_importance() -> None:
    from rustuna.optuna.importance import FanovaImportanceEvaluator

    target_mod = getattr(__import__("optuna.importance"), "importance")
    setattr(target_mod, "FanovaImportanceEvaluator", FanovaImportanceEvaluator)
