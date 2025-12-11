use rustuna_core::attr::CategoryLabel;
use serde::{Deserialize, Serialize};

use wasm_bindgen::prelude::*;

use crate::JsResult;

// We cannot define distribution classes with `wasm_bindgen` macro since
// CategoricalDistribution contains Vec<String>. So we use serde and serde-wasm-bindgen.
// See https://rustwasm.github.io/wasm-bindgen/reference/arbitrary-data-with-serde.html

#[wasm_bindgen(typescript_custom_section)]
pub const DISTRIBUTION_TYPE: &'static str = r#"
type FloatDistribution = {
  type: "FloatDistribution"
  low: number
  high: number
  step?: number
  log: boolean
}

type IntDistribution = {
  type: "IntDistribution"
  low: number
  high: number
  step?: number
  log: boolean
}

type CategoricalDistribution = {
  type: "CategoricalDistribution"
  choices: string[]
}

type Distribution =
  | FloatDistribution
  | IntDistribution
  | CategoricalDistribution
"#;

#[derive(Serialize, Deserialize)]
pub struct JsFloatDistribution {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub low: f64,
    pub high: f64,
    pub step: Option<f64>,
    pub log: bool,
}
impl JsFloatDistribution {
    pub fn new(low: f64, high: f64, step: Option<f64>, log: bool) -> Self {
        JsFloatDistribution {
            type_: "FloatDistribution",
            low,
            high,
            step,
            log,
        }
    }

    // For error handling, we define this function instead of `impl Into<JsValue> for ... {}`.
    pub fn to_js_value(&self) -> JsResult<JsValue> {
        let js_distribution = serde_wasm_bindgen::to_value(&self)
            .map_err(|e| JsError::new(&format!("Failed to serialize distribution: {e:?}")))?;
        Ok(js_distribution)
    }
}

#[derive(Serialize, Deserialize)]
pub struct JsIntDistribution {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub low: i64,
    pub high: i64,
    pub step: i64,
    pub log: bool,
}
impl JsIntDistribution {
    pub fn new(low: i64, high: i64, step: i64, log: bool) -> Self {
        JsIntDistribution {
            type_: "IntDistribution",
            low,
            high,
            step,
            log,
        }
    }

    pub fn to_js_value(&self) -> JsResult<JsValue> {
        let js_distribution = serde_wasm_bindgen::to_value(&self)
            .map_err(|e| JsError::new(&format!("Failed to serialize distribution: {e:?}")))?;
        Ok(js_distribution)
    }
}

#[derive(Serialize, Deserialize)]
pub struct JsCategoricalDistribution {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub choices: Vec<String>,
}
impl JsCategoricalDistribution {
    pub fn new(choices: Vec<CategoryLabel>) -> Self {
        let mut choices_str: Vec<String> = Vec::with_capacity(choices.len());
        for c in choices {
            choices_str.push(c.serialize());
        }
        JsCategoricalDistribution {
            type_: "CategoricalDistribution",
            choices: choices_str,
        }
    }
    pub fn to_js_value(&self) -> JsResult<JsValue> {
        let js_distribution = serde_wasm_bindgen::to_value(&self)
            .map_err(|e| JsError::new(&format!("Failed to serialize distribution: {e:?}")))?;
        Ok(js_distribution)
    }
}
