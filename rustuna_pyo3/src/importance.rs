use std::num::NonZeroUsize;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;

use fanova::{FanovaOptions, RandomForestOptions};
use rustuna_importance::fanova::get_param_importance;

use crate::study::PyStudy;

#[pyfunction]
#[pyo3(name = "get_param_importance", signature = (study))]
pub fn py_get_param_importance(study: &PyStudy) -> PyResult<Vec<Vec<f64>>> {
    let importance = get_param_importance(&study.study)
        .map_err(|_e| PyRuntimeError::new_err("Failed to get parameter importance"))?;
    Ok(importance)
}

/// This is a private API for rustuna.optuna package.
#[pyfunction]
#[pyo3(name = "_get_param_importance_from_list", signature = (features, targets, n_trees))]
pub fn py_get_param_importance_from_list(
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
    n_trees: usize,
) -> PyResult<Vec<f64>> {
    // TODO(c-bata): Try using https://github.com/PyO3/rust-numpy to make this faster.
    let features_vec = features.iter().map(|x| x.as_slice()).collect();
    let targets_vec = targets.as_slice();
    let trees = NonZeroUsize::new(n_trees).unwrap();
    let mut fanova = FanovaOptions::new()
        .random_forest(RandomForestOptions::new().trees(trees).seed(0))
        .fit(features_vec, targets_vec)
        .map_err(|_e| PyValueError::new_err("Failed to fit random forest"))?;
    let importance = (0..features.len())
        .map(|i| fanova.quantify_importance(&[i]).mean)
        .collect::<Vec<_>>();
    Ok(importance)
}
