## rustuna_js

JavaScript and WebAssembly bindings for Rustuna.

This package currently builds two wasm-bindgen outputs:

- `pkg/node/` for Node.js
- `pkg/web/` for browsers

`package.json` exposes the Node.js build as the default entrypoint and the browser build via the
`rustuna/web` subpath or the `browser` export condition.

### Node.js example

```js
import * as rustuna from "rustuna";

const study = rustuna.create_study("node-example");

study.optimize((trial) => {
  const x = trial.suggest_float("x", -10.0, 10.0);
  const y = trial.suggest_int("y", -10, 10);
  return (x - 3) ** 2 + (y + 2) ** 2;
}, 30);

console.log(study.best_trial);
```

### Browser example

For browsers, initialize the wasm module before calling exported APIs.

```html
<script type="module">
  import init, { create_study } from "./pkg/web/rustuna.js";

  await init();

  const study = create_study("browser-example");
  study.optimize((trial) => {
    const x = trial.suggest_float("x", -10.0, 10.0);
    const y = trial.suggest_int("y", -10, 10);
    return (x - 3) ** 2 + (y + 2) ** 2;
  }, 30);

  console.log(study.best_trial.toJSON());
</script>
```

A minimal browser example is available at `examples/browser/index.html`.

### Development

#### Build from source

```bash
cd rustuna_js
pnpm build
```

#### Run Node.js tests

```bash
cd rustuna_js
pnpm test
```

#### Check the browser example locally

```bash
cd rustuna_js
pnpm build
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/examples/browser/`

#### Run the TypeScript example

If `tsc` is available, `pnpm build` also emits the example bundle under `dist/`.

```bash
cd rustuna_js
pnpm build
node dist/simple_quadratic.js
```

#### Format

```bash
pnpm add -D @biomejs/biome
biome format --write examples/**/*.ts test/*.mjs
```
