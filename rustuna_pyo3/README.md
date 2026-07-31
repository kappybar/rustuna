## Rustuna

### Installation

You can install Rustuna via pip. Unlike Optuna, Rustuna doesn't have runtime dependencies, not even on NumPy. This not only eliminates concerns of version conflicts for users but also significantly speeds up imports.

```
$ pip install rustuna
```

### Example

```python
import rustuna


def objective(trial: rustuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    value = (x - 2) ** 2 + (y + 5) ** 2

    print(f"trial={trial.number}, x={x:.05}, y={y:.05}, value={value:.05}")
    return value


study = rustuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_trial)
```

## License

MIT License
