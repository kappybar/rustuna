use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pyo3::types::{PyList, PyType};
use rustuna_core::attr::{category_labels_to_attrs, get_category_labels, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;

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
    pub kind: &'static str,
}

#[pymethods]
impl PyStorage {
    #[classmethod]
    fn in_memory(_cls: &PyType) -> PyResult<Self> {
        Ok(PyStorage {
            storage: Arc::new(RwLock::new(InMemoryStorage::new())),
            kind: "in_memory",
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
        let mut guard = self.storage.write().unwrap();
        let distribution: Distribution = distribution.into();
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
        // TODO(c-bata): Add validation to detect changes of the choices.
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
        let attrs = category_labels_to_attrs(&param_name, &category_labels);
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_study_attrs(study_id, attrs)
            .map_err(err_to_exceptions)?;
        Ok(())
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
            .set_study_attrs(study_id, system_attrs)
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
            .set_study_attrs(study_id, user_attrs)
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
            .set_trial_attrs(study_id, trial_number, system_attrs)
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
            .set_trial_attrs(study_id, trial_number, user_attrs)
            .map_err(err_to_exceptions)?;
        Ok(())
    }
}
