use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rustuna_core::trial_queue::{InMemoryTrialQueue, TrialQueue};
use rustuna_core::{Error, ErrorKind, Result};
use rustuna_storage::directory_queue::DirectoryTrialQueue;
use rustuna_storage::sqlite3_queue::SQLite3TrialQueue;
use std::sync::{Arc, RwLock};

use crate::exception::err_to_exceptions;

pub struct PyObjectTrialQueue {
    obj: Py<PyAny>,
}

impl PyObjectTrialQueue {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }

    fn map_pyerr(&self, err: PyErr) -> Error {
        Python::attach(|py| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Python TrialQueueProtocol error: {}", err.value(py)),
            )
        })
    }
}

impl TrialQueue for PyObjectTrialQueue {
    fn enqueue(&mut self, trial_id: u32) -> Result<()> {
        Python::attach(|py| {
            self.obj
                .call_method1(py, "enqueue", (trial_id,))
                .map(|_| ())
                .map_err(|err| self.map_pyerr(err))
        })
    }

    fn dequeue(&mut self) -> Result<u32> {
        Python::attach(|py| {
            self.obj
                .call_method0(py, "dequeue")
                .and_then(|value| value.extract::<u32>(py))
                .map_err(|err| self.map_pyerr(err))
        })
    }
}

#[derive(Clone)]
#[pyclass(name = "TrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyTrialQueue {
    pub queue: Arc<RwLock<dyn TrialQueue>>,
}

#[derive(Clone)]
#[pyclass(name = "PyObjectTrialQueue")]
#[pyo3(module = "rustuna")]
pub struct PyPyObjectTrialQueue {
    pub queue: Arc<RwLock<PyObjectTrialQueue>>,
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
    fn sqlite3(_cls: &Bound<'_, PyType>, db_path: &str, namespace: &str) -> PyResult<Self> {
        let queue = SQLite3TrialQueue::new(db_path, namespace).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create SQLite3TrialQueue: {e:?}"))
        })?;
        Ok(PyTrialQueue {
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

#[pymethods]
impl PyPyObjectTrialQueue {
    #[new]
    fn new(trial_queue: Py<PyAny>) -> Self {
        Self {
            queue: Arc::new(RwLock::new(PyObjectTrialQueue::new(trial_queue))),
        }
    }
}
