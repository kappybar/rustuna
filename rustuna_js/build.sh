#!/bin/bash

DIR=$(cd $(dirname $0); pwd)
OUTPUT_DIR=${DIR}/pkg
NODE_OUTPUT_DIR=${OUTPUT_DIR}/node
WEB_OUTPUT_DIR=${OUTPUT_DIR}/web

set -ex

pushd ${DIR}

# Build rustuna library
cargo build --locked --target wasm32-unknown-unknown --release
mkdir -p ${NODE_OUTPUT_DIR} ${WEB_OUTPUT_DIR}
wasm-bindgen ../target/wasm32-unknown-unknown/release/rustuna.wasm --out-dir ${NODE_OUTPUT_DIR} --target nodejs
wasm-bindgen ../target/wasm32-unknown-unknown/release/rustuna.wasm --out-dir ${WEB_OUTPUT_DIR} --target web
#wasm-opt -Oz -o pkg/node/rustuna_bg.wasm pkg/node/rustuna_bg.wasm
#wasm-opt -Oz -o pkg/web/rustuna_bg.wasm pkg/web/rustuna_bg.wasm

# Examples
if command -v tsc >/dev/null 2>&1; then
  tsc --project tsconfig.examples.json
else
  echo "Skipping example compilation because tsc was not found."
fi

cat << 'EOF' > pkg/.gitignore
*
EOF
popd
