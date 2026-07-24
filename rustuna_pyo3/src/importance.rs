use std::collections::HashMap;

use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::PyResult;

use rustuna_importance::{
    get_param_importances_with, ImportanceOptions, ImportanceEvaluator, PedAnovaImportanceEvaluator,
};

use crate::study::PyStudy;

#[pyfunction]
#[pyo3(name = "get_param_importances", signature = (study, *, evaluator = None, params = None, normalize = true))]
pub fn py_get_param_importances(
    study: &PyStudy,
    evaluator: Option<&PyPedAnovaImportanceEvaluator>,
    params: Option<Vec<String>>,
    normalize: bool,
) -> PyResult<HashMap<String, f64>> {
    let options = ImportanceOptions {
        target: None,
        normalize,
        params,
    };
    let default_evaluator = PedAnovaImportanceEvaluator::default();
    let evaluator = evaluator
        .map(|wrapper| &wrapper.evaluator)
        .unwrap_or(&default_evaluator);
    let importances =
        get_param_importances_with(&study.study, evaluator, options).map_err(|err| {
            PyValueError::new_err(format!("Failed to evaluate parameter importances: {err}"))
        })?;
    Ok(importances)
}

#[pyclass(name = "PedAnovaImportanceEvaluator")]
#[pyo3(module = "rustuna.importance")]
pub struct PyPedAnovaImportanceEvaluator {
    evaluator: PedAnovaImportanceEvaluator,
}

#[pymethods]
impl PyPedAnovaImportanceEvaluator {
    #[new]
    #[pyo3(signature = (*, target_quantile = 0.1, region_quantile = 1.0, evaluate_on_local = true))]
    fn py_new(
        py: Python<'_>,
        target_quantile: f64,
        region_quantile: f64,
        evaluate_on_local: bool,
    ) -> PyResult<Self> {
        let evaluator =
            PedAnovaImportanceEvaluator::new(target_quantile, region_quantile, evaluate_on_local)
                .map_err(|err| PyValueError::new_err(err.reason))?;
        if region_quantile != 1.0 && !evaluate_on_local {
            PyErr::warn(
                py,
                &py.get_type::<PyUserWarning>(),
                pyo3::ffi::c_str!(
                    "If `evaluate_on_local` is False, `region_quantile` has no effect."
                ),
                1,
            )?;
        }
        Ok(Self { evaluator })
    }

    #[pyo3(signature = (study, params = None))]
    fn evaluate(
        &self,
        study: &PyStudy,
        params: Option<Vec<String>>,
    ) -> PyResult<HashMap<String, f64>> {
        let options = ImportanceOptions {
            target: None,
            normalize: true,
            params,
        };
        let importances =
            self.evaluator.evaluate_with(&study.study, options).map_err(|err| {
                PyValueError::new_err(format!("Failed to evaluate parameter importances: {err}"))
            })?;
        Ok(importances)
    }
}
