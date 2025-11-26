use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;

mod exceptions {
    pyo3::import_exception!(rustuna.exceptions, DuplicatedStudyError);
}

pub fn err_to_exceptions(e: rustuna_core::Error) -> PyErr {
    match e.kind {
        rustuna_core::ErrorKind::TrialNotFound => PyKeyError::new_err("Trial not found"),
        rustuna_core::ErrorKind::StudyNotFound => PyKeyError::new_err("Study not found"),
        rustuna_core::ErrorKind::DuplicatedStudy => {
            exceptions::DuplicatedStudyError::new_err("Duplicate study name")
        }
        _ => PyRuntimeError::new_err(format!("Storage Errors: {:?}", e.kind)),
    }
}
