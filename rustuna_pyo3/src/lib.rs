#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod attrs;
mod distribution;
mod exception;
mod importance;
mod pyobject_storage;
mod sampler;
mod storage;
mod study;
mod trial;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_rustuna")]
fn rustuna(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<attrs::AttrsDictView>()?;
    // trial
    m.add_class::<trial::PyTrial>()?;
    m.add_class::<trial::PyPersistedTrial>()?;
    m.add_class::<trial::PyTrialState>()?;
    // study
    m.add_function(wrap_pyfunction!(study::py_create_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::py_load_study, m)?)?;
    m.add_class::<study::PyStudy>()?;
    m.add_class::<study::PyDirection>()?;
    m.add_class::<study::PyPersistedStudy>()?;
    // distribution
    m.add_class::<distribution::PyDistribution>()?;
    // storage
    m.add_class::<storage::PyStorage>()?;
    m.add_class::<pyobject_storage::PyPyObjectStorage>()?;
    // sampler
    m.add_class::<sampler::PySampler>()?;
    m.add_class::<sampler::PySamplerContext>()?;
    // importance
    m.add_function(wrap_pyfunction!(importance::py_get_param_importance, m)?)?;
    m.add_function(wrap_pyfunction!(
        importance::py_get_param_importance_from_list,
        m
    )?)?;
    Ok(())
}
