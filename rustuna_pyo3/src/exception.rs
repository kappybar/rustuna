use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod exceptions {
    pyo3::import_exception!(rustuna.exceptions, DuplicatedStudyError);
    pyo3::import_exception!(rustuna.exceptions, TrialPruned);
    pyo3::import_exception!(rustuna.exceptions, UpdateFinishedTrialError);
    pyo3::import_exception!(rustuna.exceptions, StorageInternalError);
}

pub fn err_to_exceptions(e: rustuna_core::Error) -> PyErr {
    match e.kind {
        rustuna_core::ErrorKind::TrialNotFound => PyKeyError::new_err("Trial not found"),
        rustuna_core::ErrorKind::StudyNotFound => PyKeyError::new_err("Study not found"),
        rustuna_core::ErrorKind::AttrNotFound => PyKeyError::new_err("Attribute not found"),
        rustuna_core::ErrorKind::TrialAlreadyFinished => {
            exceptions::UpdateFinishedTrialError::new_err("Trial already finished")
        }
        rustuna_core::ErrorKind::StorageError => {
            exceptions::StorageInternalError::new_err("storage internal error")
        }
        rustuna_core::ErrorKind::DuplicatedStudy => {
            exceptions::DuplicatedStudyError::new_err("Duplicate study name")
        }
        rustuna_core::ErrorKind::IncompatibleDistribution => {
            PyValueError::new_err("Incompatible distribution for the parameter")
        }
        _ => PyRuntimeError::new_err(format!("Storage Errors: {:?}", e.kind)),
    }
}
