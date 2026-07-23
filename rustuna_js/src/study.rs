use js_sys::Function;
use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, get_best_trial, Direction, Study};
use rustuna_core::{Error, ErrorKind};
use wasm_bindgen::prelude::*;

use crate::trial::{JsPersistedTrial, JsTrial};

type JsResult<T> = Result<T, JsError>;

#[wasm_bindgen(js_name=Study)]
pub struct JsStudy(Study);
#[wasm_bindgen(js_class=Study)]
impl JsStudy {
    pub fn optimize(&self, objective: &Function, n_trials: usize) -> JsResult<()> {
        self.0
            .optimize(
                |t| {
                    let js_trial: JsTrial = t.into();
                    let result = objective.call1(objective, &JsValue::from(js_trial));
                    // TODO(c-bata): Support multi-objective optimization.
                    let val = match result {
                        Ok(js_value) => js_value
                            .as_f64()
                            .ok_or(Error::new(ErrorKind::ObjectiveError)),
                        Err(err) => Err(Error::with_reason(
                            ErrorKind::ObjectiveError,
                            format!("Objective function failed: {err:?}"),
                        )),
                    }?;
                    Ok(vec![val])
                },
                n_trials,
            )
            .map_err(|err| JsError::new(&format!("{err:?}")))
    }

    #[wasm_bindgen(getter)]
    pub fn best_trial(&self) -> JsResult<JsPersistedTrial> {
        let number = get_best_trial(&self.0).map_err(|e| JsError::new(&format!("{e:?}")))?;
        let (trials, study_attrs) = {
            let mut guard = self.0.storage.write().map_err(|e| {
                JsError::new(&format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let trials = guard
                .get_trials(self.0.id)
                .map_err(|e| JsError::new(&format!("Failed to get trials: {:?}", e.kind)))?
                .clone();
            let study_attrs = guard
                .get_study(self.0.id)
                .map_err(|e| JsError::new(&format!("Failed to get study: {:?}", e.kind)))?
                .attrs
                .clone();
            (trials, study_attrs)
        };
        let trial = JsPersistedTrial::new(
            trials[number as usize]
                .clone()
                .ok_or_else(|| JsError::new("Trial is missing"))?,
            study_attrs,
        );
        Ok(trial)
    }
}
impl From<Study> for JsStudy {
    fn from(item: Study) -> Self {
        JsStudy(item)
    }
}

#[wasm_bindgen(js_name=create_study)]
pub fn js_create_study(study_name: String) -> JsResult<JsStudy> {
    let storage = InMemoryStorage::new();
    let directions = vec![Direction::Minimize];
    let study = create_study(&study_name, storage, RandomSampler::new(), directions);
    match study {
        Ok(study) => Ok(study.into()),
        Err(err) => Err(JsError::new(&format!("{err:?}"))),
    }
}
