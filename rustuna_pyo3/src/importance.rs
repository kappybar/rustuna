use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyResult;

use rustuna_importance::{get_param_importances_with, ImportanceOptions, PedAnovaImportanceEvaluator};

use crate::study::PyStudy;

#[pyfunction]
#[pyo3(name = "get_param_importances", signature = (study, *, evaluator = None, params = None, normalize = true))]
pub fn py_get_param_importances(study: &PyStudy, evaluator: Option<PyPedAnovaImportanceEvaluator>, params: Option<Vec<String>>, normalize: bool) -> PyResult<HashMap<String,f64>> {
    let opts = ImportanceOptions {
        target: None,
        normalize,
        params,
    };
    let evaluator = evaluator.map(|wrapper| wrapper.evaluator).unwrap_or(PedAnovaImportanceEvaluator::default());
    let importances = get_param_importances_with(&study.study, &evaluator, opts)
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to get parameter importances: {err:?}")))?;
    Ok(importances)
}

#[pyclass(name = "PedAnovaImportanceEvaluator")]
#[pyo3(module = "rustuna.importance")]
pub struct PyPedAnovaImportanceEvaluator {
    pub evaluator: PedAnovaImportanceEvaluator,
}
