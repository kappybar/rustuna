#!/bin/bash

DIR=$(cd $(dirname $0); pwd)
OUTPUT_DIR=${DIR}/pkg

set -ex

pushd ${DIR}

# Build rustuna library
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen ../target/wasm32-unknown-unknown/release/rustuna.wasm --out-dir ${OUTPUT_DIR} --target nodejs
#wasm-opt -Oz -o pkg/rustuna_bg.wasm pkg/rustuna_bg.wasm

# Examples
tsc --project tsconfig.examples.json

cat << 'EOF' > pkg/.gitignore
*
EOF
popd
