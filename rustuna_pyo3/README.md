## rustuna_pyo3

### Example

```python
import rustuna as optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    value = (x - 2) ** 2 + (y + 5) ** 2

    print(f"trial={trial.number}, x={x:.05}, y={y:.05}, value={value:.05}")
    return value

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_trial)
```

### Contributing

#### Setup

```
$ uv sync --group dev
```

#### Installation

```
$ cd rustuna_pyo3/
$ uv run python python_examples/simple_quadratic.py
```

#### Debugging with rust-gdb

```
$ RUST_BACKTRACE=1 maturin develop
$ rust-gdb --args python python_examples/simple_quadratic.py
```

#### Build wheel packages

```
$ docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --manifest-path rustuna_pyo3/Cargo.toml
📦 Built wheel for CPython 3.8 to /io/target/wheels/rustuna-0.1.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
$ ls target/wheels/
rustuna-0.1.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

#### Test

```
$ uv run pytest tests/
```

#### Lint

```
$ uv run ruff format .
$ uv run ruff check . --fix
```
