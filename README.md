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

<details>
<summary>Optuna compatible interfaces</summary>

Though Rustuna offers Optuna-like APIs, full compatibility with Optuna isn't guaranteed.
Nevertheless, the `rustuna.optuna` subpackage provides an Optuna compatible interface.
The following code is an example of calculating the importance of hyperparameters with Rustuna.

```python
import optuna
from rustuna.optuna.importance import FanovaImportanceEvaluator


def objective(trial: optuna.Trial) -> float:
    ...

storage = optuna.storages.InMemoryStorage()
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=sampler, storage=storage)

importance = optuna.importance.get_param_importances(
    study, evaluator=FanovaImportanceEvaluator()
)
print(importance)
```

We also offer a monkey patch for users who wish to speed up programs while preserving the original code.
Despite Rustuna being in its early development stages, if unforeseeable circumstances arise halting its development, you can easily revert back to Optuna simply by removing this code.

```python
from rustuna.optuna import monkey
monkey.patch_all()  # Replace optuna's fanova with rustuna's fanova

from optuna.importance import FanovaImportanceEvaluator
...
```

</details>

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

### C

```c
#include <stdio.h>

#include "rustuna.h"

int main(void) {
  TunaTrial *trial;
  double x;
  double values[1];

  TunaDirection direction[1] = {TunaDirectionMinimize};
  TunaSampler *sampler = tuna_new_tpe_sampler();
  TunaStudy *study = tuna_create_study("test_study", *sampler, direction, 1);
  for (int i = 0; i < 30; i++) {
    trial = tuna_ask(study);
    if (trial == NULL) {
      return 1;
    }
    tuna_suggest_float(trial, "x1", -10.0, 10.0, &x);
    values[0] = x * x;
    tuna_tell(study, trial->number, values, 1);
    printf("  value=%f (x1 = %f)\n", values[0], x);
  }
  return 0;
}
````

### C++

```cpp
#include <cmath>
#include <iostream>

#include "rustuna.hpp"

int main() {
  rustuna::Study study("test_study", {TunaDirectionMinimize});
  for (int i = 0; i < 100; i++) {
    rustuna::Trial trial = study.ask();
    double x = trial.suggest_float("x", -10.0, 10.0);
    int y = trial.suggest_int("y", -10, 10);
    std::string z = trial.suggest_categorical("z", {"foo", "bar", "baz"});

    double objective = pow(x - 3, 2) + pow(y + 5, 2);
    study.tell(trial, {objective});

    std::cout objective << " (x=" << x << ", y=" << y << ", z=" << z << ")" << std::endl;
  }
  return 0;
}
```

## Crates

- `rustuna_core` : Core components of Rustuna.
- `rustuna_js` : The JavaScript binding.
- `rustuna_pyo3` : The Python binding.
- `rustuna_c` : The C-API and the C++ wrapper.
- `rustuna_samplers` : A collection of Rustuna samplers.
- `rustuna_importance` : A collection of hyperparameter importance evaluators.
