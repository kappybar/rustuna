use wasm_bindgen::prelude::*;

type JsResult<T> = Result<T, JsError>;

mod distribution;
mod study;
mod trial;
