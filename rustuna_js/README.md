## rustuna_js

### Example

```js
import * as wasm from './pkg/rustuna_wasm.js';

const study = wasm.create_study("test");

const objective = (trial) => {
    const x = trial.suggest_float("x", -10.0, 10.0);
    const y = trial.suggest_int("y", -10, 10);
    const z = trial.suggest_categorical("z", ["foo", "bar"]);

    const value = (x - 5) ** 2 + (y + 2) ** 2
    console.log(`x: ${x}, y: ${y}, z: ${z}, value: ${value}`)
    return value
}
study.optimize(objective, 10)
console.log(study.best_trial())
console.log("best_trial.value: ", study.best_trial().value)
```

### Contributing

#### Build from Source

```
$ cd rustuna_js/
$ ./build.sh
```

#### Run examples

```
$ ./build.sh
$ node dist/simple_quadratic.js
```

#### Format

```
$ npm i -g @biomejs/biome
$ biome format --write examples/*.ts test/*.mjs
```

#### Test

```
$ node --test
```

