# Rustuna Documentation

Welcome to Rustuna documentation. Rustuna is a hyperparameter optimization framework written in Rust, inspired by [Optuna](https://optuna.org/).

## Features

- Fast optimization powered by Rust
- Python bindings for easy integration
- Compatible with Optuna's core concepts

## Quick Start

```python
import rustuna as optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    value = (x - 2) ** 2 + (y + 5) ** 2
    return value

study = optuna.create_study()
study.optimize(objective, n_trials=1000)
print(study.best_trial)
```

## Installation

```bash
pip install rustuna
```

For more details, see the [Getting Started](tutorial/getting-started.md) guide.
