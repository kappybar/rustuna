# Contributing to Rustuna

> [!NOTE]
> Rustuna is not currently accepting external pull requests. The Optuna maintainers
> have received a growing number of LLM-generated pull requests in recent years,
> which has created a substantial review burden. In addition, Rustuna is still at an
> early stage of development and may undergo significant architectural changes.
> Accepting external pull requests at this stage could make it difficult to iterate
> quickly.
>
> We hope to accept external contributions in the future. In the meantime, feedback
> and suggestions are welcome through [GitHub Issues](https://github.com/optuna/rustuna/issues).

## Repository layout

- `rustuna_core`: Core components and abstractions.
- `rustuna_sampler`: Sampler implementations.
- `rustuna_storage`: Storage implementations.
- `rustuna_importance`: Hyperparameter importance evaluators.
- `rustuna_pyo3`: Python bindings.
- `rustuna_js`: JavaScript and WebAssembly bindings.

Unless otherwise noted, run the commands below from the repository root.

## Rust development

### Build

Build the entire workspace:

```console
$ cargo build --workspace
```

To build a single crate, specify its package name:

```console
$ cargo build -p rustuna_core
```

### Test

Run the Rust test suite with all features enabled:

```console
$ cargo test --all-features
```

Some storage tests require Python and Optuna. Set up the Python development
environment before running these ignored tests:

```console
$ cd rustuna_pyo3
$ uv sync --group dev
$ cd ..
$ source rustuna_pyo3/.venv/bin/activate
$ cargo test -p rustuna_storage -- --ignored
```

### Format and lint

Format the Rust code:

```console
$ cargo fmt --all
```

Run Clippy with the same options used in CI:

```console
$ cargo clippy --workspace --lib --bins --tests --examples --all-features -- -D warnings
```

### Run an example

```console
$ cargo run -p rustuna_sampler --example quadratic
```

### Profile an example

With [`cargo-flamegraph`](https://github.com/flamegraph-rs/flamegraph) installed,
generate a flame graph for the sampler example:

```console
$ CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph -p rustuna_sampler --example quadratic
```

## Python development

The Python bindings are located in `rustuna_pyo3`.

### Set up the development environment

```console
$ cd rustuna_pyo3
$ uv sync --group dev
$ uv run maturin develop
```

### Test

```console
$ uv run pytest tests/
```

### Format, lint, and type-check

Format the Python code:

```console
$ uv run ruff format .
```

Run Ruff and mypy:

```console
$ uv run ruff check .
$ uv run mypy rustuna/ python_examples/ tests/
```

### Debug with rust-gdb

```console
$ source .venv/bin/activate
$ RUST_BACKTRACE=1 maturin develop
$ rust-gdb --args python python_examples/simple_quadratic.py
```

## JavaScript and WebAssembly development

The JavaScript and WebAssembly bindings are located in `rustuna_js`. The build
produces two wasm-bindgen packages:

- `pkg/node/` for Node.js
- `pkg/web/` for browsers

The Node.js package is the default entry point. The browser package is available
through the `rustuna/web` subpath and the `browser` export condition.

### Build

The build requires the `wasm32-unknown-unknown` Rust target and
`wasm-bindgen-cli`.

```console
$ cd rustuna_js
$ pnpm build
```

### Test

```console
$ pnpm test
```

### Format and lint

With [Biome](https://biomejs.dev/) installed, format the source files:

```console
$ biome format --write examples/*.ts test/*.mjs
```

Run the same checks used in CI:

```console
$ biome ci examples/*.ts test/*.mjs
$ tsc --noEmit --project tsconfig.examples.json
```

### Run the examples

To run the Node.js example, build the package and execute the generated bundle:

```console
$ pnpm build
$ node dist/simple_quadratic.js
```

The TypeScript compiler must be available when running `pnpm build`; otherwise,
the example bundle is not generated.

To serve the browser example locally:

```console
$ pnpm build
$ python3 -m http.server 8000
```

Then open <http://localhost:8000/examples/browser/>.
