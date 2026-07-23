use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustuna_core::trial_queue::TrialQueue;
use rustuna_storage::directory_queue::DirectoryTrialQueue;
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "DirectoryTrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyDirectoryTrialQueue {
    pub queue: Arc<RwLock<DirectoryTrialQueue>>,
}

#[pymethods]
impl PyDirectoryTrialQueue {
    #[new]
    fn py_new(base_dir: &str) -> PyResult<Self> {
        let queue = DirectoryTrialQueue::new(base_dir).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create DirectoryTrialQueue: {e:?}"))
        })?;
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
