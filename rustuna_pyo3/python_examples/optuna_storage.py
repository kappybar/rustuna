from __future__ import annotations

import optuna

import rustuna
from rustuna.converter import ToRustunaStorage


def objective(trial: rustuna.Trial | optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    value = x**2
    return value


def main() -> None:
    # Sample 10 trials with Optuna
    optuna_storage = optuna.storages.RDBStorage("sqlite://")
    optuna_study = optuna.create_study(storage=optuna_storage)
    optuna_study.optimize(objective, n_trials=10)
    print(optuna_study.best_trial)

    # Resume the optimization with Rustuna
    rustuna_storage = ToRustunaStorage(optuna_storage)
    rustuna_study = rustuna.load_study(
        study_name=optuna_study.study_name, storage=rustuna_storage
    )
    rustuna_study.optimize(objective, n_trials=10)
    print(rustuna_study.best_trial)

    print(f"{len(rustuna_study.trials)=}")


if __name__ == "__main__":
    main()
