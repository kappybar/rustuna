use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat};
use pyo3::Python;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use rustuna_core::sampler::RandomSampler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use rustuna_core::attr::AttrKey;
use rustuna_core::sampler::Sampler;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::{
    create_study_with_arc, get_best_trial, get_pareto_front, Direction, PersistedStudy, Study,
};
use rustuna_core::trial::TrialStateValues;
use rustuna_samplers::tpe::TpeSampler;

use crate::attrs::pyobj_to_attrs;
use crate::exception::err_to_exceptions;
use crate::pyobject_storage::PyObjectStorage;
use crate::sampler::{PyObjectSampler, PySampler};
use crate::storage::PyStorage;
use crate::trial::{PyPersistedTrial, PyTrial, PyTrialState};

#[pyfunction]
#[pyo3(name = "create_study", signature = (*, study_name = None, storage = None, sampler = None, direction = None, directions = None))]
pub fn py_create_study(
    study_name: Option<String>,
    storage: Option<PyObject>,
    sampler: Option<PyObject>,
    direction: Option<String>,
    directions: Option<Vec<String>>,
) -> PyResult<PyStudy> {
    let study_name = match study_name {
        Some(s) => s,
        None => "default".to_string(), // TODO(c-bata): Generate random name with uuid.
    };
    let storage: PyResult<Arc<RwLock<dyn Storage>>> = match storage {
        Some(storage_obj) => Python::with_gil(|py| {
            let storage_ref = storage_obj.bind(py);
            if storage_ref.is_instance_of::<PyStorage>() {
                let storage = storage_ref.extract::<PyStorage>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PyStorage: {e:?}"))
                })?;
                Ok(storage.storage)
            } else {
                let is_distributed = Python::with_gil(|py| {
                    storage_obj
                        .getattr(py, "is_distributed")?
                        .extract::<bool>(py)
                })?;
                let mut storage = PyObjectStorage::new(storage_obj, is_distributed);
                storage.sync_studies(true).map_err(err_to_exceptions)?;
                let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(storage));
                Ok(storage)
            }
        }),
        None => Ok(Arc::new(RwLock::new(InMemoryStorage::new()))),
    };
    let directions = convert_directions(direction, directions)?;
    let is_multi_objective = directions.len() > 1;
    let study =
        create_study_with_arc(&study_name, storage?, directions).map_err(err_to_exceptions)?;
    let sampler: PyResult<Arc<Mutex<dyn Sampler>>> = match sampler {
        Some(sampler_obj) => Python::with_gil(|py| {
            let sampler_ref = sampler_obj.bind(py);
            if sampler_ref.is_instance_of::<PySampler>() {
                let sampler = sampler_ref.extract::<PySampler>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PySampler: {e:?}"))
                })?;
                Ok(sampler.sampler)
            } else {
                let sampler: Arc<Mutex<dyn Sampler>> =
                    Arc::new(Mutex::new(PyObjectSampler::new(sampler_obj)));
                Ok(sampler)
            }
        }),
        None => {
            let sampler: Arc<Mutex<dyn Sampler>> = if is_multi_objective {
                Arc::new(Mutex::new(RandomSampler::new()))
            } else {
                Arc::new(Mutex::new(TpeSampler::new()))
            };
            Ok(sampler)
        }
    };
    Ok(PyStudy {
        study,
        sampler: sampler?,
    })
}

#[pyfunction]
#[pyo3(name = "load_study", signature = (study_name, storage, *, sampler = None))]
pub fn py_load_study(
    study_name: String,
    storage: PyObject,
    sampler: Option<PyObject>,
) -> PyResult<PyStudy> {
    let storage: PyResult<Arc<RwLock<dyn Storage>>> = Python::with_gil(|py| {
        let storage_ref = storage.bind(py);
        if storage_ref.is_instance_of::<PyStorage>() {
            let storage = storage_ref.extract::<PyStorage>().map_err(|e| {
                PyValueError::new_err(format!("Failed to extract PyStorage: {e:?}"))
            })?;
            let storage: Arc<RwLock<dyn Storage>> = storage.storage;
            Ok(storage)
        } else {
            let mut storage = PyObjectStorage::new(storage, true);
            storage.sync_studies(true).map_err(err_to_exceptions)?;
            let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(storage));
            Ok(storage)
        }
    });
    let storage = storage?;
    let mut guard = storage
        .write()
        .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
    let (study_id, directions) = guard
        .get_studies()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get the studies: {:?}", e.kind)))?
        .iter()
        .find(|s| s.name == study_name)
        .map(|s| (s.id, s.directions.clone()))
        .ok_or(PyRuntimeError::new_err(format!(
            "Study {study_name} not found"
        )))?;
    drop(guard);
    let is_multi_objective = directions.len() > 1;
    let study = Study::new(study_id, study_name, directions, storage);
    let sampler: PyResult<Arc<Mutex<dyn Sampler>>> = match sampler {
        Some(sampler_obj) => Python::with_gil(|py| {
            let sampler_ref = sampler_obj.bind(py);
            if sampler_ref.is_instance_of::<PySampler>() {
                let sampler = sampler_ref.extract::<PySampler>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PySampler: {e:?}"))
                })?;
                Ok(sampler.sampler)
            } else {
                let sampler: Arc<Mutex<dyn Sampler>> =
                    Arc::new(Mutex::new(PyObjectSampler::new(sampler_obj)));
                Ok(sampler)
            }
        }),
        None => {
            let sampler: Arc<Mutex<dyn Sampler>> = if is_multi_objective {
                Arc::new(Mutex::new(RandomSampler::new()))
            } else {
                Arc::new(Mutex::new(TpeSampler::new()))
            };
            Ok(sampler)
        }
    };
    Ok(PyStudy {
        study,
        sampler: sampler?,
    })
}

#[pyclass(name = "Study")]
#[pyo3(module = "rustuna")]
pub struct PyStudy {
    pub study: Study,
    sampler: Arc<Mutex<dyn Sampler>>,
}
#[allow(non_local_definitions)]
#[pymethods]
impl PyStudy {
    #[new]
    #[pyo3(signature = (study_id, name, directions, storage, sampler))]
    fn py_new(
        study_id: u32,
        name: String,
        directions: Vec<PyDirection>,
        storage: PyStorage,
        sampler: PySampler,
    ) -> Self {
        let directions: Vec<Direction> = directions.into_iter().map(|d| d.into()).collect();
        let study = Study::new(study_id, name, directions, storage.storage);
        PyStudy {
            study,
            sampler: sampler.sampler,
        }
    }

    #[pyo3(signature = (objective, n_trials))]
    pub fn optimize(&mut self, objective: PyObject, n_trials: usize) -> PyResult<()> {
        for _ in 0..n_trials {
            // Ask a trial
            let sampler = self.sampler.clone();
            let rs_trial = self
                .study
                .ask(sampler)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to ask a trial {e:?}.")))?;
            let trial_number = rs_trial.number;
            let trial: PyTrial = rs_trial.into();

            // Call an objective function
            let result: PyResult<Vec<f64>> = Python::with_gil(|py| {
                let val = objective.call1(py, (trial,))?;
                let val_ref = val.bind(py);
                if val_ref.is_instance_of::<PyFloat>() {
                    let val = val_ref.extract::<f64>().map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to extract f64: {e:?}"))
                    })?;
                    Ok(vec![val])
                } else {
                    val_ref.extract::<Vec<f64>>().map_err(|e| {
                        PyRuntimeError::new_err(format!("Objective function must return either float or tuple[float]. error={e:?}"))
                    })
                }
            });

            // Tell
            match result {
                Ok(val) => {
                    self.study
                        .tell(trial_number, TrialStateValues::Complete(val))
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;
                }
                Err(e) => {
                    self.study
                        .tell(trial_number, TrialStateValues::Fail)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn ask(&mut self) -> PyResult<PyTrial> {
        let trial: PyTrial = self
            .study
            .ask(self.sampler.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to ask a trial: {:?}", e.kind)))?
            .into();
        Ok(trial)
    }

    #[pyo3(signature = (number, values = None, state = None))]
    pub fn tell(
        &mut self,
        number: u32,
        values: Option<PyObject>,
        state: Option<PyTrialState>,
    ) -> PyResult<()> {
        let state_values = match (state, values) {
            (None, None) => Err(PyValueError::new_err(
                "Either state or values must be specified",
            )),
            (Some(PyTrialState::RUNNING), _) => Err(PyValueError::new_err(
                "Cannot tell running trials with values",
            )),
            (Some(PyTrialState::WAITING), _) => Err(PyValueError::new_err(
                "Cannot tell waiting trials with values",
            )),
            (Some(PyTrialState::FAIL), _) => Ok(TrialStateValues::Fail),
            (Some(PyTrialState::PRUNED), _) => Ok(TrialStateValues::Pruned),
            (Some(PyTrialState::COMPLETE), None) => Err(PyValueError::new_err(
                "values must be specified when state is COMPLETE",
            )),
            (Some(PyTrialState::COMPLETE), Some(values)) => {
                let state_values: PyResult<TrialStateValues> = Python::with_gil(|py| {
                    let val = values.bind(py);
                    if val.is_instance_of::<PyFloat>() {
                        let val = val.extract::<f64>()?;
                        Ok(TrialStateValues::Complete(vec![val]))
                    } else {
                        let val = val.extract::<Vec<f64>>()?;
                        Ok(TrialStateValues::Complete(val))
                    }
                });
                state_values
            }
            (None, Some(values)) => {
                let state_values: PyResult<TrialStateValues> = Python::with_gil(|py| {
                    let val = values.bind(py);
                    if val.is_instance_of::<PyFloat>() {
                        let val = val.extract::<f64>()?;
                        Ok(TrialStateValues::Complete(vec![val]))
                    } else {
                        let val = val.extract::<Vec<f64>>()?;
                        Ok(TrialStateValues::Complete(val))
                    }
                });
                state_values
            }
        };
        self.study
            .tell(number, state_values?)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;
        Ok(())
    }

    #[getter]
    pub fn best_trial(&mut self) -> PyResult<PyPersistedTrial> {
        let trial_number = get_best_trial(&self.study).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get the best trial: {:?}", e.kind))
        })?;

        let mut guard = self.study.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        let trial_id = guard
            .get_trial_id_from_study_id_trial_number(self.study.id, trial_number)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get trial id: {:?}", e.kind))
            })?;
        let trial = guard
            .get_trial(trial_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trial: {:?}", e.kind)))?;
        Ok(PyPersistedTrial::from_storage(
            self.study.storage.clone(),
            trial,
        ))
    }

    #[getter]
    pub fn trials(&mut self) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let trials: Vec<PyPersistedTrial> = trials_vec
            .iter()
            .map(|t| PyPersistedTrial::from_storage(self.study.storage.clone(), t))
            .collect();
        Ok(trials)
    }

    // TODO(c-bata): Add user_attrs property method and set_user_attrs() method.

    #[getter]
    pub fn id(&self) -> u32 {
        self.study.id
    }

    #[getter]
    pub fn directions(&self) -> Vec<PyDirection> {
        self.study
            .directions
            .iter()
            .map(|d| d.clone().into())
            .collect()
    }

    #[getter]
    pub fn best_trials(&mut self) -> PyResult<Vec<PyPersistedTrial>> {
        let pareto_front_numbers = get_pareto_front(&self.study).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get the pareto front: {:?}", e.kind))
        })?;
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let best_trials = pareto_front_numbers
            .iter()
            .map(|n| {
                PyPersistedTrial::from_storage(self.study.storage.clone(), &trials_vec[*n as usize])
            })
            .collect();
        Ok(best_trials)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let type_obj = slf.get_type();
        let class_name = type_obj.name()?;
        Ok(format!("{}({})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "id={} name={} directions={:?}",
            self.study.id,
            self.study.name,
            self.directions()
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[pyclass(name = "StudyDirection", eq, eq_int)]
#[pyo3(module = "rustuna")]
pub enum PyDirection {
    #[pyo3(name = "MINIMIZE")]
    Minimize,
    #[pyo3(name = "MAXIMIZE")]
    Maximize,
}
impl From<Direction> for PyDirection {
    fn from(item: Direction) -> Self {
        match item {
            Direction::Minimize => PyDirection::Minimize,
            Direction::Maximize => PyDirection::Maximize,
        }
    }
}
impl From<PyDirection> for Direction {
    fn from(val: PyDirection) -> Self {
        match val {
            PyDirection::Minimize => Direction::Minimize,
            PyDirection::Maximize => Direction::Maximize,
        }
    }
}

fn convert_directions(
    direction: Option<String>,
    directions: Option<Vec<String>>,
) -> PyResult<Vec<Direction>> {
    if direction.is_some() && directions.is_some() {
        Err(PyValueError::new_err(
            "Cannot specify both `direction` and `directions`",
        ))?;
    };
    let direction = match direction {
        Some(d) => match d.as_str() {
            "minimize" => Direction::Minimize,
            "maximize" => Direction::Maximize,
            _ => Err(PyValueError::new_err(
                "Invalid direction. Please specify either `minimize` or `maximize`",
            ))?,
        },
        None => Direction::Minimize,
    };
    let directions = match directions {
        Some(ds) => ds
            .into_iter()
            .map(|d| match d.as_str() {
                "minimize" => Ok(Direction::Minimize),
                "maximize" => Ok(Direction::Maximize),
                _ => Err(PyValueError::new_err(
                    "Invalid direction. Please specify either `minimize` or `maximize`",
                )),
            })
            .collect(),
        None => Ok(vec![direction]),
    }?;
    Ok(directions)
}

#[derive(Debug, Clone)]
#[pyclass(name = "PersistedStudy")]
#[pyo3(module = "rustuna", get_all, set_all)]
pub struct PyPersistedStudy {
    pub id: u32,
    pub name: String,
    pub directions: Vec<PyDirection>,
    pub user_attrs: HashMap<String, String>,
    pub system_attrs: HashMap<String, String>,
}
impl From<PersistedStudy> for PyPersistedStudy {
    fn from(item: PersistedStudy) -> Self {
        let cap = std::cmp::min(item.attrs.len() / 2, 1);
        let mut user_attrs: HashMap<String, String> = HashMap::with_capacity(cap);
        let mut system_attrs: HashMap<String, String> = HashMap::with_capacity(cap);

        for (key, val) in item.attrs {
            match key {
                AttrKey::User(k) => {
                    user_attrs.insert(k.to_string(), val);
                }
                AttrKey::System(k) => {
                    system_attrs.insert(k.to_string(), val);
                }
            }
        }
        let directions = item.directions.into_iter().map(|d| d.into()).collect();
        PyPersistedStudy {
            id: item.id,
            name: item.name,
            directions,
            user_attrs,
            system_attrs,
        }
    }
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyPersistedStudy {
    #[new]
    #[pyo3(signature = (id, name, directions, user_attrs=None, system_attrs=None))]
    pub fn py_new(
        id: u32,
        name: String,
        directions: Vec<PyDirection>,
        user_attrs: Option<HashMap<String, String>>,
        system_attrs: Option<HashMap<String, String>>,
    ) -> Self {
        PyPersistedStudy {
            id,
            name,
            directions,
            user_attrs: user_attrs.unwrap_or_default(),
            system_attrs: system_attrs.unwrap_or_default(),
        }
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let type_obj = slf.get_type();
        let class_name = type_obj.name()?;
        Ok(format!("{}({:?})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "id={} name={} user_attrs={:?} system_attrs={:?}",
            self.id, self.name, self.user_attrs, self.system_attrs
        ))
    }
}

pub fn pyobject_to_persisted_study(study: &Bound<'_, PyAny>) -> PyResult<PersistedStudy> {
    let study_id = study.getattr("id")?.extract::<u32>()?;
    let name = study.getattr("name")?.extract::<String>()?;
    let directions = study.getattr("directions")?.extract::<Vec<PyDirection>>()?;
    let directions: Vec<Direction> = directions.iter().map(|d| d.clone().into()).collect();

    let user_attrs = study.getattr("user_attrs")?;
    let system_attrs = study.getattr("system_attrs")?;
    if !user_attrs.is_instance_of::<PyDict>() || !system_attrs.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(
            "user_attrs and system_attrs must be a dict",
        ));
    }
    let attrs = pyobj_to_attrs(&user_attrs, &system_attrs)?;
    Ok(PersistedStudy::new_with_attrs(
        study_id, name, directions, attrs,
    ))
}
