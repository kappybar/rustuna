use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

use crate::trial_queue::python::PythonTrialQueueAdapter;

pub mod directory;
pub mod inmemory;
pub mod python;
pub mod sqlite3;

#[derive(Clone)]
#[pyclass(name = "PyObjectTrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyPyObjectTrialQueue {
    pub queue: Arc<RwLock<PythonTrialQueueAdapter>>,
}

#[pymethods]
impl PyPyObjectTrialQueue {
    #[new]
    fn new(trial_queue: Py<PyAny>) -> Self {
        Self {
            queue: Arc::new(RwLock::new(PythonTrialQueueAdapter::new(trial_queue))),
        }
    }
}
