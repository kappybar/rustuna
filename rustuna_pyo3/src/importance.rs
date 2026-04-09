use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyResult;

use rustuna_importance::fanova::get_param_importance;

use crate::study::PyStudy;

#[pyfunction]
#[pyo3(name = "get_param_importance", signature = (study))]
pub fn py_get_param_importance(study: &PyStudy) -> PyResult<Vec<Vec<f64>>> {
    let importance = get_param_importance(&study.study)
        .map_err(|_e| PyRuntimeError::new_err("Failed to get parameter importance"))?;
    Ok(importance)
}
