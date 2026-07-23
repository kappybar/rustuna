use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustuna_core::trial_queue::TrialQueue;
use rustuna_storage::sqlite3_queue::SQLite3TrialQueue;
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "SQLite3TrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PySQLite3TrialQueue {
    pub queue: Arc<RwLock<SQLite3TrialQueue>>,
}

#[pymethods]
impl PySQLite3TrialQueue {
    #[new]
    fn py_new(db_path: &str, namespace: &str) -> PyResult<Self> {
        let queue = SQLite3TrialQueue::new(db_path, namespace).map_err(|error| {
            PyRuntimeError::new_err(format!(
                "Failed to create SQLite3TrialQueue: {}",
                error.reason
            ))
        })?;
        Ok(Self {
            queue: Arc::new(RwLock::new(queue)),
        })
    }

    fn enqueue(&self, trial_id: u32) -> PyResult<()> {
        let mut guard = self.queue.write().map_err(|error| {
            PyRuntimeError::new_err(format!("Failed to acquire trial queue lock: {error}"))
        })?;
        guard.enqueue(trial_id).map_err(err_to_exceptions)
    }

    fn dequeue(&self) -> PyResult<u32> {
        let mut guard = self.queue.write().map_err(|error| {
            PyRuntimeError::new_err(format!("Failed to acquire trial queue lock: {error}"))
        })?;
        guard.dequeue().map_err(err_to_exceptions)
    }
}
