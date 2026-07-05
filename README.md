# Rustuna

A faster Optuna implementation in Rust, featuring Python, JavaScript, and C bindings.

## Getting Started

### Python

You can install Rustuna via pip. Unlike Optuna, Rustuna doesn't have runtime dependencies, not even on NumPy.
This not only eliminates concerns of version conflicts for users but also significantly speeds up imports.

```
$ pip install rustuna
```

Here's a basic use case:

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

### JavaScript

```
$ npm i --save rustuna
```

```js
import * as rustuna from "rustuna.js";

const study = rustuna.create_study("test");

const objective = (trial: rustuna.Trial) => {
	const x = trial.suggest_float("x", -10.0, 10.0);
	const y = trial.suggest_int("y", -10, 10);
	const z = trial.suggest_categorical("z", ["foo", "bar"]);

	const value = (x - 5) ** 2 + (y + 2) ** 2;
	console.log(`x: ${x}, y: ${y}, z: ${z}, value: ${value}`);
	return value;
};
study.optimize(objective, 10);

console.log(study.best_trial);
```

### Rust

```rust
use std::sync::{Arc, Mutex};

use rustuna_core::objective_single::get_best_trial;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{Direction, create_study};
use rustuna_core::Result;
use rustuna_samplers::tpe::TpeSampler;

fn main() -> Result<()> {
    let storage = InMemoryStorage::new();
    let mut study = create_study("simple-quadratic", storage, vec![Direction::Minimize])?;

    let sampler = Arc::new(Mutex::new(TpeSampler::new()));
    study.optimize(
        |mut t| {
            let x = t.suggest_float("x", 0.0, 10.0)?;
            let y = t.suggest_float("y", 0.0, 10.0)?;
            let value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
            println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
            Ok(vec![value])
        },
        sampler,
        100,
    )?;

    println!("Best trial: {:?}", get_best_trial(&study)?);
    Ok(())
}
```

## Crates

- `rustuna_core` : Core components of Rustuna.
- `rustuna_js` : The JavaScript binding.
- `rustuna_pyo3` : The Python binding.
- `rustuna_samplers` : A collection of Rustuna samplers.
- `rustuna_storage` : A collection of Rustuna storage implementations.
- `rustuna_importance` : A collection of hyperparameter importance evaluators.
