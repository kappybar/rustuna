use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rustuna_core::trial_queue::{InMemoryTrialQueue, TrialQueue};
use rustuna_storages::directory_queue::DirectoryTrialQueue;
use rustuna_storages::sqlite3_queue::SQLite3TrialQueue;
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "TrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyTrialQueue {
    pub queue: Arc<RwLock<dyn TrialQueue>>,
}

#[pymethods]
impl PyTrialQueue {
    #[classmethod]
    fn in_memory(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let queue = InMemoryTrialQueue::new();
        Ok(PyTrialQueue {
            queue: Arc::new(RwLock::new(queue)),
        })
    }

    #[classmethod]
    fn directory(_cls: &Bound<'_, PyType>, base_dir: &str) -> PyResult<Self> {
        let queue = DirectoryTrialQueue::new(base_dir).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create DirectoryTrialQueue: {e:?}"))
        })?;
        Ok(PyTrialQueue {
            queue: Arc::new(RwLock::new(queue)),
        })
    }

    #[classmethod]
    fn sqlite3(_cls: &Bound<'_, PyType>, db_path: &str, study_id: u32) -> PyResult<Self> {
        let queue = SQLite3TrialQueue::new(db_path, study_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create SQLite3TrialQueue: {e:?}"))
        })?;
        Ok(PyTrialQueue {
            queue: Arc::new(RwLock::new(queue)),
        })
    }

    fn push(&self, trial_id: u32) -> PyResult<()> {
        let mut guard = self
            .queue
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire queue lock"))?;
        guard.push(trial_id).map_err(err_to_exceptions)
    }

    fn pop(&self) -> PyResult<u32> {
        let mut guard = self
            .queue
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire queue lock"))?;
        guard.pop().map_err(err_to_exceptions)
    }
}
