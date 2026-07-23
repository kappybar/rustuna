use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustuna_core::trial_queue::{InMemoryTrialQueue, TrialQueue};
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "InMemoryTrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyInMemoryTrialQueue {
    pub queue: Arc<RwLock<InMemoryTrialQueue>>,
}

#[pymethods]
impl PyInMemoryTrialQueue {
    #[new]
    fn py_new() -> PyResult<Self> {
        let queue = InMemoryTrialQueue::new();
        Ok(Self {
            queue: Arc::new(RwLock::new(queue)),
        })
    }

    fn enqueue(&self, trial_id: u32) -> PyResult<()> {
        let mut guard = self
            .queue
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire queue lock"))?;
        guard.enqueue(trial_id).map_err(err_to_exceptions)
    }

    fn dequeue(&self) -> PyResult<u32> {
        let mut guard = self
            .queue
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire queue lock"))?;
        guard.dequeue().map_err(err_to_exceptions)
    }
}
