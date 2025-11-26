import time

import rustuna as optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    value = (x - 2) ** 2 + (y + 5) ** 2

    print(f"trial={trial.number}, x={x:.05}, y={y:.05}, value={value:.05}")
    return value


def main() -> None:
    # By default, Rustuna uses InMemory storage and TPE sampler.
    start = time.time()
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    elapsed = time.time() - start
    print(f"Done: elapsed={elapsed:.03} sec")

    # Print best trial
    print(study.best_trial)

    # Print parameter importances
    print(optuna.get_param_importance(study))


if __name__ == "__main__":
    main()
