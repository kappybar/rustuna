use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyResult;

use rustuna_importance::fanova::get_param_importance;

use crate::study::PyStudy;

#[pyfunction]
#[pyo3(name = "get_param_importances", signature = (study, *, evaluator = None, params = None, normalize = true))]
pub fn py_get_param_importances(study: &PyStudy, evaluator: Option<PedAnovaImportanceEvaluator>, params: Option<Vec<String>>, normalize: bool) -> PyResult<HashMap<String,f64>> {
    let opts = ImportanceOptions {
        target: None,
        normalize,
        params,
    };
    let importance = get_param_importances_with(&study.study, &evaluator.unwrap_or(PedAnovaImportanceEvaluator::default()), opts)
        .map_err(|_e| PyRuntimeError::new_err("Failed to get parameter importance"))?;
    Ok(importance)
}
