use js_sys::Array;
use rustuna_core::attr::{get_category_labels, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::trial::{PersistedTrial, Trial, TrialStateValues};
use wasm_bindgen::prelude::*;

use crate::distribution::{JsCategoricalDistribution, JsFloatDistribution, JsIntDistribution};
use crate::JsResult;

fn category_label_to_js_value(label: &CategoryLabel) -> JsValue {
    match label {
        CategoryLabel::Float(f) => JsValue::from_f64(*f),
        CategoryLabel::Int(i) => JsValue::from_f64(*i as f64),
        CategoryLabel::String(s) => JsValue::from_str(s),
        CategoryLabel::Bool(b) => JsValue::from_bool(*b),
        CategoryLabel::None => JsValue::null(),
    }
}

#[wasm_bindgen(js_name=Trial)]
pub struct JsTrial(Trial);
impl From<Trial> for JsTrial {
    fn from(item: Trial) -> Self {
        JsTrial(item)
    }
}
#[wasm_bindgen(js_class=Trial)]
impl JsTrial {
    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> JsResult<f64> {
        match self.0.suggest_float(name, low, high) {
            Ok(value) => Ok(value),
            Err(err) => Err(JsError::new(&format!("{err:?}"))),
        }
    }
    // Use i32 since ECMAScript's number type can represent the values in [-2^53, 2^53].
    // Another note is that there is no difference between suggest_int and suggest_float
    // in terms of JavaScript types, since there is only one number type in JavaScript.
    pub fn suggest_int(&mut self, name: &str, low: i32, high: i32) -> JsResult<i32> {
        match self.0.suggest_int(name, low as i64, high as i64) {
            Ok(value) => Ok(value as i32),
            Err(err) => Err(JsError::new(&format!("{err:?}"))),
        }
    }
    /// Suggest a value for a categorical parameter.
    /// The choices argument should be an array of string, number, boolean, or null.
    pub fn suggest_categorical(&mut self, name: &str, choices: Array) -> JsResult<JsValue> {
        let mut category_labels: Vec<CategoryLabel> = Vec::with_capacity(choices.length() as usize);
        for i in 0..choices.length() {
            let c = choices.get(i);
            if c.is_string() {
                category_labels
                    .push(CategoryLabel::String(c.as_string().ok_or(JsError::new(
                        "Failed to read string category value",
                    ))?));
                continue;
            } else if c.is_null() {
                category_labels.push(CategoryLabel::None);
                continue;
            }
            let js_ty = c
                .js_typeof()
                .as_string()
                .ok_or(JsError::new("Unsupported category value type"))?;
            if js_ty == "number" {
                category_labels
                    .push(CategoryLabel::Float(c.as_f64().ok_or(JsError::new(
                        "Failed to read numeric category value",
                    ))?));
            } else if js_ty == "boolean" {
                category_labels
                    .push(CategoryLabel::Bool(c.as_bool().ok_or(JsError::new(
                        "Failed to read boolean category value",
                    ))?));
            } else {
                return Err(JsError::new("Unsupported category value type"));
            }
        }

        match self.0.suggest_categorical_enum(name, &category_labels) {
            Ok(value) => Ok(category_label_to_js_value(value)),
            Err(err) => Err(JsError::new(&format!("{err:?}"))),
        }
    }

    pub fn set_user_attr(&mut self, key: &str, value: String) -> JsResult<()> {
        match self.0.set_user_attr(key, value) {
            Ok(_) => Ok(()),
            Err(err) => Err(JsError::new(&format!("{err:?}"))),
        }
    }
}

#[wasm_bindgen(typescript_custom_section)]
pub const PERSISTED_TRIAL_TYPE: &'static str = r#"
type TrialState = "Running" | "Complete" | "Pruned" | "Fail" | "Waiting"

type TrialParam = {
  name: string
  internal_value: number
  external_value: number
  distribution: Distribution
}

type PersistedTrialJSON = {
  number: number
  state: TrialState
  values: number[] | null
  user_attrs: { key: string, value: string }[]
  params: TrialParam[]
}

export class PersistedTrial {
  toJSON(): PersistedTrialJSON;
  toString(): string;
  free(): void;
  readonly number: number;
  readonly params: TrialParam[];
  readonly state: TrialState;
  readonly user_attrs: { key: string, value: string }[];
  readonly values: number[];
}
"#;

#[wasm_bindgen(js_name=PersistedTrial, inspectable, skip_typescript)]
pub struct JsPersistedTrial(PersistedTrial, Attrs);
impl JsPersistedTrial {
    pub fn new(trial: PersistedTrial, study_attrs: Attrs) -> Self {
        JsPersistedTrial(trial, study_attrs)
    }
}
#[wasm_bindgen(js_class=PersistedTrial)]
impl JsPersistedTrial {
    #[wasm_bindgen(getter)]
    pub fn number(&self) -> u32 {
        self.0.number
    }

    #[wasm_bindgen(getter)]
    pub fn state(&self) -> String {
        match self.0.state_values {
            TrialStateValues::Running => String::from("Running"),
            TrialStateValues::Complete(_) => String::from("Complete"),
            TrialStateValues::Fail => String::from("Fail"),
            TrialStateValues::Pruned => String::from("Pruned"),
            TrialStateValues::Waiting => String::from("Waiting"),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> JsValue {
        let values = js_sys::Array::new();
        match &self.0.state_values {
            TrialStateValues::Complete(v) => {
                v.iter().for_each(|x| {
                    values.push(&JsValue::from_f64(*x));
                });
                values.into()
            }
            _ => JsValue::null(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn user_attrs(&self) -> JsResult<JsValue> {
        let attrs = js_sys::Array::new();
        let user_attrs = self.0.attrs.iter().filter_map(|(key, value)| {
            if let rustuna_core::attr::AttrKey::User(key) = key {
                Some((key.to_string(), value))
            } else {
                None
            }
        });
        for (key, value) in user_attrs {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("key"), &JsValue::from_str(&key))
                .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("value"), &JsValue::from_str(value))
                .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
            attrs.push(&obj);
        }
        Ok(attrs.into())
    }

    #[wasm_bindgen(getter)]
    pub fn params(&self) -> JsResult<JsValue> {
        let params = js_sys::Array::new();
        for (param_name, internal_value) in self.0.internal_params.iter() {
            let param_obj = js_sys::Object::new();

            js_sys::Reflect::set(
                &param_obj,
                &JsValue::from_str("name"),
                &JsValue::from_str(param_name),
            )
            .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
            js_sys::Reflect::set(
                &param_obj,
                &JsValue::from_str("internal_value"),
                &JsValue::from_f64(*internal_value),
            )
            .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;

            let distribution = self
                .0
                .distributions
                .get(param_name)
                .ok_or(JsError::new("Failed to get a distribution"))?;
            match distribution {
                Distribution::Float {
                    low,
                    high,
                    step,
                    log,
                } => {
                    js_sys::Reflect::set(
                        &param_obj,
                        &JsValue::from_str("external_value"),
                        &JsValue::from_f64(*internal_value),
                    )
                    .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;

                    js_sys::Reflect::set(
                        &param_obj,
                        &JsValue::from_str("distribution"),
                        &JsFloatDistribution::new(*low, *high, *step, *log).to_js_value()?,
                    )
                    .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
                }
                Distribution::Int {
                    low,
                    high,
                    step,
                    log,
                } => {
                    js_sys::Reflect::set(
                        &param_obj,
                        &JsValue::from_str("external_value"),
                        &JsValue::from_f64(*internal_value),
                    )
                    .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;

                    js_sys::Reflect::set(
                        &param_obj,
                        &JsValue::from_str("distribution"),
                        &JsIntDistribution::new(*low, *high, *step, *log).to_js_value()?,
                    )
                    .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
                }
                Distribution::Categorical { cardinality } => {
                    let labels = match get_category_labels(&self.1, param_name, *cardinality) {
                        Some(labels) => labels,
                        None => {
                            // This branch is unreachable unless trying to load the params
                            // sampled by suggest_categorical<T> in Rust.
                            let mut labels: Vec<CategoryLabel> = Vec::with_capacity(*cardinality);
                            for i in 0..*cardinality {
                                labels.push(CategoryLabel::Int(i as i64));
                            }
                            labels
                        }
                    };

                    let c = match labels.get(*internal_value as usize) {
                        Some(c) => category_label_to_js_value(c),
                        None => {
                            return Err(JsError::new("Failed to get a category value"));
                        }
                    };
                    js_sys::Reflect::set(&param_obj, &JsValue::from_str("external_value"), &c)
                        .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;

                    js_sys::Reflect::set(
                        &param_obj,
                        &JsValue::from_str("distribution"),
                        &JsCategoricalDistribution::new(labels).to_js_value()?,
                    )
                    .map_err(|e| JsError::new(&format!("Failed to set a property: {e:?}")))?;
                }
            }
            params.push(&param_obj);
        }
        Ok(params.into())
    }
}
