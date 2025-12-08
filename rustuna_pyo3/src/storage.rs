use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pyo3::types::{PyList, PyType};
use rustuna_core::attr::{category_labels_to_attrs, get_category_labels, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_storages::cache::CachedStorage;
use rustuna_storages::optuna::OptunaCompatibleStorage;
use rustuna_storages::sqlite3::SQLite3Storage;

use crate::attrs::{pyobj_to_system_attrs, pyobj_to_user_attrs};
use crate::distribution::{category_label_to_pyobject, pyobject_to_category_label, PyDistribution};
use crate::exception::err_to_exceptions;
use crate::study::{PyDirection, PyPersistedStudy};
use crate::trial::{PyPersistedTrial, PyTrialState};

#[derive(Clone)]
#[pyclass(name = "Storage")]
#[pyo3(module = "rustuna")]
pub struct PyStorage {
    pub storage: Arc<RwLock<dyn Storage>>,
    pub optuna_compatible: Option<Arc<RwLock<dyn OptunaCompatibleStorage>>>,
    pub kind: &'static str,
}

#[pymethods]
impl PyStorage {
    #[classmethod]
    fn in_memory(_cls: &PyType) -> PyResult<Self> {
        Ok(PyStorage {
            storage: Arc::new(RwLock::new(InMemoryStorage::new())),
            optuna_compatible: None,
            kind: "in_memory",
        })
    }

    #[classmethod]
    #[pyo3(name = "sqlite3", signature = (file_path, *, create_database = false))]
    fn sqlite3(_cls: &PyType, file_path: &str, create_database: bool) -> PyResult<Self> {
        let backend = SQLite3Storage::new(file_path).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to open the SQLite3 file: {e:?}"))
        })?;
        if create_database {
            backend.create_database().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create the database: {e:?}"))
            })?;
        }

        let arc_storage = Arc::new(RwLock::new(CachedStorage::new(Box::new(backend))));
        Ok(PyStorage {
            storage: arc_storage.clone(),
            optuna_compatible: Some(arc_storage),
            kind: "sqlite3",
        })
    }

    fn create_new_study(
        &mut self,
        study_name: String,
        directions: Vec<PyDirection>,
    ) -> PyResult<PyPersistedStudy> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let directions: Vec<Direction> = directions.iter().map(|d| d.clone().into()).collect();
        let study = guard
            .create_new_study(&study_name, directions)
            .map_err(err_to_exceptions)?;
        Ok(study.clone().into())
    }

    fn delete_study(&mut self, study_id: u32) -> PyResult<()> {
        let mut guard = self.storage.write().unwrap();
        guard.delete_study(study_id).map_err(err_to_exceptions)?;
        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> PyResult<PyPersistedTrial> {
        let mut guard = self.storage.write().unwrap();
        let trial = guard
            .create_new_trial(study_id)
            .map_err(err_to_exceptions)?;
        Ok(PyPersistedTrial::new(trial.clone(), Default::default()))
    }

    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: String,
        distribution: PyDistribution,
        value: f64,
    ) -> PyResult<()> {
        let category_labels = distribution.category_labels.clone();
        let distribution: Distribution = distribution.into();

        if let Some(labels) = category_labels {
            self.set_category_labels_internal(study_id, name.clone(), labels)?;
        }

        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_trial_param(study_id, trial_number, &name, &distribution, value)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: String,
        choices: Vec<PyObject>,
    ) -> PyResult<()> {
        let category_labels = Python::with_gil(|py| {
            let mut labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
            for choice in choices {
                match pyobject_to_category_label(py, choice) {
                    Ok(label) => labels.push(label),
                    Err(e) => return Err(e),
                }
            }
            Ok(labels)
        })?;
        self.set_category_labels_internal(study_id, param_name, category_labels)
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: String,
        cardinality: usize,
    ) -> PyResult<PyObject> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
        Python::with_gil(
            |py| match get_category_labels(&study.attrs, &param_name, cardinality) {
                Some(labels) => {
                    let mut elements: Vec<PyObject> = Vec::with_capacity(cardinality);
                    for i in 0..cardinality {
                        let c = labels.get(i).ok_or(PyValueError::new_err(
                            "Internal representation of categorical value is out of range",
                        ))?;
                        elements.push(category_label_to_pyobject(py, c));
                    }
                    let choices = PyList::new(py, &elements);
                    Ok(choices.into())
                }
                None => Ok(py.None()),
            },
        )
    }

    #[pyo3(signature = (study_id, trial_number, state, values=None))]
    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut guard = self.storage.write().unwrap();

        let state_values = match state {
            PyTrialState::COMPLETE => {
                let values = values.ok_or(PyValueError::new_err(
                    "values must be specified when state is COMPLETE",
                ))?;
                TrialStateValues::Complete(values)
            }
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::WAITING => TrialStateValues::Waiting,
            PyTrialState::FAIL => TrialStateValues::Fail,
        };
        guard
            .set_trial_state_values(study_id, trial_number, state_values)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn get_studies(&mut self) -> PyResult<Vec<PyPersistedStudy>> {
        let mut guard = self.storage.write().unwrap();
        let studies = guard
            .get_studies()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get studies: {:?}", e.kind)))?;
        Ok(studies.iter().map(|s| s.clone().into()).collect())
    }

    fn get_study(&mut self, study_id: u32) -> PyResult<PyPersistedStudy> {
        let mut guard = self.storage.write().unwrap();
        let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
        Ok(study.clone().into())
    }

    fn get_trials(&mut self, study_id: u32) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self.storage.write().unwrap();
        let study_attrs = {
            let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
            study.attrs.clone()
        };
        let trials = guard.get_trials(study_id).map_err(err_to_exceptions)?;
        // TODO(c-bata): Filter category_labels attrs and clone them only.
        let py_trials: Vec<PyPersistedTrial> = trials
            .iter()
            .map(|t| PyPersistedTrial::new(t.clone(), study_attrs.clone()))
            .collect();
        Ok(py_trials)
    }

    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> PyResult<PyPersistedTrial> {
        let mut guard = self.storage.write().unwrap();
        let trial = guard
            .get_trial(study_id, trial_number)
            .map_err(err_to_exceptions)?
            .clone();
        let study_attrs = guard
            .get_study(study_id)
            .map_err(err_to_exceptions)?
            .attrs
            .clone();
        Ok(PyPersistedTrial::new(trial, study_attrs))
    }

    fn set_study_system_attrs(&mut self, study_id: u32, attrs: PyObject) -> PyResult<()> {
        let system_attrs = Python::with_gil(|py| {
            let attrs = attrs.as_ref(py);
            pyobj_to_system_attrs(attrs)
        })?;
        let mut guard = self.storage.write().unwrap();
        guard
            .set_study_attrs(study_id, system_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_study_user_attrs(&mut self, study_id: u32, attrs: PyObject) -> PyResult<()> {
        let user_attrs = Python::with_gil(|py| {
            let attrs = attrs.as_ref(py);
            pyobj_to_user_attrs(attrs)
        })?;
        let mut guard = self.storage.write().unwrap();
        guard
            .set_study_attrs(study_id, user_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_trial_system_attrs(
        &mut self,
        study_id: u32,
        trial_number: u32,
        attrs: PyObject,
    ) -> PyResult<()> {
        let system_attrs = Python::with_gil(|py| {
            let attrs = attrs.as_ref(py);
            pyobj_to_system_attrs(attrs)
        })?;
        let mut guard = self.storage.write().unwrap();
        guard
            .set_trial_attrs(study_id, trial_number, system_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_trial_user_attrs(
        &mut self,
        study_id: u32,
        trial_number: u32,
        attrs: PyObject,
    ) -> PyResult<()> {
        let user_attrs = Python::with_gil(|py| {
            let attrs = attrs.as_ref(py);
            pyobj_to_user_attrs(attrs)
        })?;
        let mut guard = self.storage.write().unwrap();
        guard
            .set_trial_attrs(study_id, trial_number, user_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> PyResult<u32> {
        let optuna_storage = self.optuna_compatible.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("This storage does not support Optuna-compatible operations")
        })?;
        let mut guard = optuna_storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trial_id = guard
            .get_trial_id_from_study_id_trial_number(study_id, trial_number)
            .map_err(err_to_exceptions)?;
        Ok(trial_id)
    }

    fn get_study_id_trial_number_from_trial_id(&mut self, trial_id: u32) -> PyResult<(u32, u32)> {
        let optuna_storage = self.optuna_compatible.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("This storage does not support Optuna-compatible operations")
        })?;
        let mut guard = optuna_storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let (study_id, trial_number) = guard
            .get_study_id_trial_number_from_trial_id(trial_id)
            .map_err(err_to_exceptions)?;
        Ok((study_id, trial_number))
    }

    fn set_trial_intermediate_value(
        &mut self,
        trial_id: u32,
        step: u32,
        intermediate_value: f64,
    ) -> PyResult<()> {
        let optuna_storage = self.optuna_compatible.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("This storage does not support Optuna-compatible operations")
        })?;
        let mut guard = optuna_storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let mut intermediate_values = std::collections::HashMap::new();
        intermediate_values.insert(step, intermediate_value);
        guard
            .set_trial_intermediate_values(trial_id, intermediate_values)
            .map_err(err_to_exceptions)?;
        Ok(())
    }
}

impl PyStorage {
    fn set_category_labels_internal(
        &mut self,
        study_id: u32,
        param_name: String,
        category_labels: Vec<CategoryLabel>,
    ) -> PyResult<()> {
        let attrs = category_labels_to_attrs(&param_name, &category_labels);
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        match guard.set_study_attrs(study_id, attrs, true) {
            Ok(_) => Ok(()),
            Err(e) => {
                if matches!(e.kind, rustuna_core::ErrorKind::AttrOverwriteNotAllowed) {
                    let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
                    let existing_labels =
                        get_category_labels(&study.attrs, &param_name, category_labels.len());
                    if let Some(existing) = existing_labels {
                        if existing == category_labels {
                            return Ok(());
                        }
                    }
                    return Err(PyValueError::new_err(format!(
                        "Cannot overwrite category labels for parameter '{}'",
                        param_name
                    )));
                }
                Err(err_to_exceptions(e))
            }
        }
    }
}
