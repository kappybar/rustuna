use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyIterator, PyType};
use pyo3::{PyTypeInfo, Python};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use rustuna_core::sampler::RandomSampler;
use rustuna_core::ErrorKind;
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
use crate::attrs::{convert_pydict_to_fixed_params, pyobj_to_attrs_with_kind, AttrKind};
use crate::exception::err_to_exceptions;
use crate::pyobject_storage::PyObjectStorage;
use crate::sampler::{PyObjectSampler, PySampler};
use crate::storage::PyStorage;
use crate::trial::{PyPersistedTrial, PyTrial, PyTrialState};
use crate::trial_queue::PyTrialQueue;

type SharedStorage = Arc<RwLock<dyn Storage>>;
type SharedTrialQueue = Arc<RwLock<dyn rustuna_core::trial_queue::TrialQueue>>;

mod py_exceptions {
    pyo3::import_exception!(rustuna.exceptions, TrialPruned);
}

fn normalize_catch(py: Python<'_>, catch: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
    let Some(catch) = catch else {
        return Ok(Vec::new());
    };
    let base_exception = py.import("builtins")?.getattr("BaseException")?;

    if let Ok(catch_type) = catch.cast::<PyType>() {
        if catch_type.is_subclass(&base_exception)? {
            return Ok(vec![catch.clone().unbind()]);
        }
    }

    let iter = PyIterator::from_object(catch)?;
    let mut normalized = Vec::new();
    for item in iter {
        let item = item?;
        let item_type = item.cast::<PyType>().map_err(|_| {
            PyTypeError::new_err(
                "The catch argument must be an exception class or an iterable of exception classes.",
            )
        })?;
        if !item_type.is_subclass(&base_exception)? {
            return Err(PyTypeError::new_err(
                "The catch argument must be an exception class or an iterable of exception classes.",
            ));
        }
        normalized.push(item.unbind());
    }
    Ok(normalized)
}

fn objective_result_to_values(val: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if val.is_instance_of::<PyFloat>() || val.is_instance_of::<PyInt>() {
        let val = val
            .extract::<f64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract f64: {e:?}")))?;
        Ok(vec![val])
    } else {
        let iter = PyIterator::from_object(&val).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Objective function must return either int, float or tuple[int | float]. error={e:?}"
            ))
        })?;
        let mut vals = Vec::new();
        for item in iter {
            let item = item.map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Objective function must return either int, float or tuple[int | float]. error={e:?}"
                ))
            })?;
            let v = if item.is_instance_of::<PyInt>() {
                item.extract::<i64>()? as f64
            } else {
                item.extract::<f64>()?
            };
            vals.push(v);
        }
        Ok(vals)
    }
}

fn matches_any_exception(py: Python<'_>, err: &PyErr, catch: &[Py<PyAny>]) -> PyResult<bool> {
    for exc_type in catch {
        if err.matches(py, exc_type.bind(py))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn into_trial_queue_pyobj(
    py: Python<'_>,
    trial_queue: Option<PyTrialQueue>,
) -> PyResult<(SharedTrialQueue, Py<PyAny>)> {
    match trial_queue {
        Some(trial_queue) => {
            let queue = trial_queue.queue.clone();
            let py_trial_queue = Py::new(py, trial_queue)?.into_any();
            Ok((queue, py_trial_queue))
        }
        None => {
            let trial_queue = PyTrialQueue {
                queue: Arc::new(RwLock::new(
                    rustuna_core::trial_queue::InMemoryTrialQueue::new(),
                )),
            };
            let queue = trial_queue.queue.clone();
            let py_trial_queue = Py::new(py, trial_queue)?.into_any();
            Ok((queue, py_trial_queue))
        }
    }
}

#[pyfunction]
#[pyo3(name = "create_study", signature = (*, study_name = None, storage = None, sampler = None, direction = None, directions = None, load_if_exists = false, trial_queue = None))]
pub fn py_create_study(
    study_name: Option<String>,
    storage: Option<Py<PyAny>>,
    sampler: Option<Py<PyAny>>,
    direction: Option<String>,
    directions: Option<Vec<String>>,
    load_if_exists: bool,
    trial_queue: Option<PyTrialQueue>,
) -> PyResult<PyStudy> {
    let study_name = match study_name {
        Some(s) => s,
        None => "default".to_string(), // TODO(c-bata): Generate random name with uuid.
    };
    let (storage_arc, storage_pyobj): (Arc<RwLock<dyn Storage>>, Py<PyAny>) = match storage {
        Some(storage_obj) => Python::attach(|py| {
            let storage_pyobj = storage_obj.clone_ref(py);
            let storage_ref = storage_obj.bind(py);
            if storage_ref.is_instance_of::<PyStorage>() {
                let storage = storage_ref.extract::<PyStorage>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PyStorage: {e:?}"))
                })?;
                Ok::<_, PyErr>((storage.storage, storage_pyobj))
            } else {
                let mut storage = PyObjectStorage::new(storage_obj);
                storage.sync_studies(true).map_err(err_to_exceptions)?;
                let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(storage));
                Ok((storage, storage_pyobj))
            }
        })?,
        None => {
            let arc: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(InMemoryStorage::new()));
            let py_storage = PyStorage {
                storage: arc.clone(),
                optuna_compatible: None,
                kind: "in_memory",
            };
            let storage_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, py_storage)?.into_any())
            })?;
            (arc, storage_pyobj)
        }
    };
    let directions = convert_directions(direction, directions)?;
    let study = match create_study_with_arc(&study_name, storage_arc.clone(), directions) {
        Ok(study) => study,
        Err(err) => {
            if !load_if_exists || !matches!(err.kind, ErrorKind::DuplicatedStudy) {
                return Err(err_to_exceptions(err));
            }
            let mut guard = storage_arc
                .write()
                .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
            let (study_id, directions) = guard
                .get_studies()
                .map_err(err_to_exceptions)?
                .iter()
                .find(|s| s.name == study_name)
                .map(|s| (s.id, s.directions.clone()))
                .ok_or(PyRuntimeError::new_err(format!(
                    "Study {study_name} not found"
                )))?;
            drop(guard);
            Study::new(
                study_id,
                study_name.clone(),
                directions,
                storage_arc.clone(),
            )
        }
    };
    let (trial_queue_arc, trial_queue_pyobj) =
        Python::attach(|py| into_trial_queue_pyobj(py, trial_queue))?;
    let study = Study::with_queue(
        study.id,
        study.name,
        study.directions,
        study.storage,
        trial_queue_arc,
    );
    let is_multi_objective = study.directions.len() > 1;
    let (sampler_arc, sampler_pyobj): (Arc<Mutex<dyn Sampler>>, Py<PyAny>) = match sampler {
        Some(sampler_obj) => Python::attach(|py| {
            let sampler_pyobj = sampler_obj.clone_ref(py);
            let sampler_ref = sampler_obj.bind(py);
            if sampler_ref.is_instance_of::<PySampler>() {
                let sampler = sampler_ref.extract::<PySampler>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PySampler: {e:?}"))
                })?;
                Ok::<_, PyErr>((sampler.sampler, sampler_pyobj))
            } else {
                let sampler: Arc<Mutex<dyn Sampler>> =
                    Arc::new(Mutex::new(PyObjectSampler::new(sampler_obj)));
                Ok((sampler, sampler_pyobj))
            }
        })?,
        None => {
            let arc: Arc<Mutex<dyn Sampler>> = if is_multi_objective {
                Arc::new(Mutex::new(RandomSampler::new()))
            } else {
                Arc::new(Mutex::new(TpeSampler::new()))
            };
            let py_sampler = PySampler {
                sampler: arc.clone(),
                kind: "tpe",
            };
            let sampler_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, py_sampler)?.into_any())
            })?;
            (arc, sampler_pyobj)
        }
    };
    Ok(PyStudy {
        study,
        sampler: sampler_arc,
        storage_pyobj,
        sampler_pyobj,
        trial_queue_pyobj,
    })
}

#[pyfunction]
#[pyo3(name = "load_study", signature = (study_name, storage, *, sampler = None, trial_queue = None))]
pub fn py_load_study(
    study_name: String,
    storage: Py<PyAny>,
    sampler: Option<Py<PyAny>>,
    trial_queue: Option<PyTrialQueue>,
) -> PyResult<PyStudy> {
    let storage_pyobj = Python::attach(|py| storage.clone_ref(py));
    let storage: PyResult<Arc<RwLock<dyn Storage>>> = Python::attach(|py| {
        let storage_ref = storage.bind(py);
        if storage_ref.is_instance_of::<PyStorage>() {
            let storage = storage_ref.extract::<PyStorage>().map_err(|e| {
                PyValueError::new_err(format!("Failed to extract PyStorage: {e:?}"))
            })?;
            let storage: Arc<RwLock<dyn Storage>> = storage.storage;
            Ok(storage)
        } else {
            let mut storage = PyObjectStorage::new(storage);
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
    let (trial_queue_arc, trial_queue_pyobj) =
        Python::attach(|py| into_trial_queue_pyobj(py, trial_queue))?;
    let study = Study::with_queue(study_id, study_name, directions, storage, trial_queue_arc);
    let (sampler_arc, sampler_pyobj): (Arc<Mutex<dyn Sampler>>, Py<PyAny>) = match sampler {
        Some(sampler_obj) => Python::attach(|py| {
            let sampler_pyobj = sampler_obj.clone_ref(py);
            let sampler_ref = sampler_obj.bind(py);
            if sampler_ref.is_instance_of::<PySampler>() {
                let sampler = sampler_ref.extract::<PySampler>().map_err(|e| {
                    PyValueError::new_err(format!("Failed to extract PySampler: {e:?}"))
                })?;
                Ok::<_, PyErr>((sampler.sampler, sampler_pyobj))
            } else {
                let sampler: Arc<Mutex<dyn Sampler>> =
                    Arc::new(Mutex::new(PyObjectSampler::new(sampler_obj)));
                Ok((sampler, sampler_pyobj))
            }
        })?,
        None => {
            let arc: Arc<Mutex<dyn Sampler>> = if is_multi_objective {
                Arc::new(Mutex::new(RandomSampler::new()))
            } else {
                Arc::new(Mutex::new(TpeSampler::new()))
            };
            let py_sampler = PySampler {
                sampler: arc.clone(),
                kind: "tpe",
            };
            let sampler_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, py_sampler)?.into_any())
            })?;
            (arc, sampler_pyobj)
        }
    };
    Ok(PyStudy {
        study,
        sampler: sampler_arc,
        storage_pyobj,
        sampler_pyobj,
        trial_queue_pyobj,
    })
}

#[pyclass(name = "Study")]
#[pyo3(module = "rustuna")]
pub struct PyStudy {
    pub study: Study,
    sampler: Arc<Mutex<dyn Sampler>>,
    storage_pyobj: Py<PyAny>,
    sampler_pyobj: Py<PyAny>,
    trial_queue_pyobj: Py<PyAny>,
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
    ) -> PyResult<Self> {
        let directions: Vec<Direction> = directions.into_iter().map(|d| d.into()).collect();
        let storage_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(Py::new(py, storage.clone())?.into_any())
        })?;
        let sampler_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(Py::new(py, sampler.clone())?.into_any())
        })?;
        let trial_queue = PyTrialQueue {
            queue: Arc::new(RwLock::new(
                rustuna_core::trial_queue::InMemoryTrialQueue::new(),
            )),
        };
        let trial_queue_arc = trial_queue.queue.clone();
        let trial_queue_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(Py::new(py, trial_queue)?.into_any())
        })?;
        let study = Study::with_queue(study_id, name, directions, storage.storage, trial_queue_arc);
        Ok(PyStudy {
            study,
            sampler: sampler.sampler,
            storage_pyobj,
            sampler_pyobj,
            trial_queue_pyobj,
        })
    }

    #[pyo3(signature = (objective, n_trials, catch = None))]
    pub fn optimize(
        &self,
        objective: Py<PyAny>,
        n_trials: usize,
        catch: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let catch = Python::attach(|py| normalize_catch(py, catch.as_ref().map(|c| c.bind(py))))?;
        for _ in 0..n_trials {
            let sampler = self.sampler.clone();
            let rs_trial = self
                .study
                .ask(sampler)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to ask a trial {e:?}.")))?;
            let trial_number = rs_trial.number;

            let result: PyResult<Vec<f64>> = Python::attach(|py| {
                let trial = PyTrial::new(rs_trial, self.storage_pyobj.clone_ref(py));
                let val = objective.call1(py, (trial,))?;
                objective_result_to_values(val.bind(py).clone())
            });

            match result {
                Ok(val) => {
                    self.study
                        .tell(trial_number, TrialStateValues::Complete(val))
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;
                }
                Err(e) => {
                    let (state, should_reraise) = Python::attach(|py| -> PyResult<_> {
                        if e.matches(py, py_exceptions::TrialPruned::type_object(py))? {
                            Ok((TrialStateValues::Pruned, false))
                        } else {
                            let should_reraise = !matches_any_exception(py, &e, &catch)?;
                            Ok((TrialStateValues::Fail, should_reraise))
                        }
                    })?;
                    self.study.tell(trial_number, state).map_err(|err| {
                        PyRuntimeError::new_err(format!("Failed to tell: {err:?}"))
                    })?;
                    if should_reraise {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn ask(&self) -> PyResult<PyTrial> {
        let trial = self
            .study
            .ask(self.sampler.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to ask a trial: {:?}", e.kind)))?;
        let trial = Python::attach(|py| PyTrial::new(trial, self.storage_pyobj.clone_ref(py)));
        Ok(trial)
    }

    #[pyo3(signature = (number, values = None, state = None))]
    pub fn tell(
        &self,
        number: u32,
        values: Option<Py<PyAny>>,
        state: Option<PyTrialState>,
    ) -> PyResult<PyPersistedTrial> {
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
            (Some(PyTrialState::COMPLETE), Some(values)) => Python::attach(|py| {
                objective_result_to_values(values.bind(py).clone()).map(TrialStateValues::Complete)
            }),
            (None, Some(values)) => Python::attach(|py| {
                objective_result_to_values(values.bind(py).clone()).map(TrialStateValues::Complete)
            }),
        };
        self.study
            .tell(number, state_values?)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;

        let mut guard = self.study.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        let trial_id = guard
            .get_trial_id_from_study_id_trial_number(self.study.id, number)
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

    #[pyo3(signature = (params, user_attrs = None))]
    pub fn enqueue_trial(
        &self,
        params: &Bound<'_, PyDict>,
        user_attrs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let fixed_params = convert_pydict_to_fixed_params(params)?;
        let user_attrs_opt = user_attrs
            .map(|d| pyobj_to_attrs_with_kind(d.as_any(), AttrKind::User))
            .transpose()?;
        self.study
            .enqueue_trial(fixed_params, user_attrs_opt)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    pub fn add_trial(&self, trial: &Bound<'_, PyPersistedTrial>) -> PyResult<()> {
        // Extract the underlying PersistedTrial
        let persisted_trial = trial.borrow().with_trial(|t| Ok(t.clone()))?;

        // Call the core implementation
        self.study
            .add_trial(persisted_trial)
            .map_err(err_to_exceptions)?;

        Ok(())
    }

    #[pyo3(signature = (key, value))]
    pub fn set_user_attr(&self, key: String, value: String) -> PyResult<()> {
        let mut attrs = rustuna_core::attr::Attrs::new();
        attrs.insert(AttrKey::User(key.into()), value);
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_study_attrs(self.study.id, attrs, false)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set user attrs: {:?}", e.kind))
            })?;
        Ok(())
    }

    #[getter]
    pub fn best_trial(&self) -> PyResult<PyPersistedTrial> {
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

    #[getter(trials)]
    pub fn py_trials(&self) -> PyResult<Vec<PyPersistedTrial>> {
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

    #[pyo3(name = "get_trials", signature = (*, states = None))]
    pub fn py_get_trials(
        &self,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let trials: Vec<PyPersistedTrial> = match states {
            Some(states) => trials_vec
                .iter()
                .filter(|trial| states.contains(&PyTrialState::from(trial.state_values.clone())))
                .map(|trial| PyPersistedTrial::from_storage(self.study.storage.clone(), trial))
                .collect(),
            None => trials_vec
                .iter()
                .map(|trial| PyPersistedTrial::from_storage(self.study.storage.clone(), trial))
                .collect(),
        };
        Ok(trials)
    }

    #[getter]
    pub fn user_attrs(&self) -> PyResult<HashMap<String, String>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let study = guard
            .get_study(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get study: {:?}", e.kind)))?;
        let mut user_attrs = HashMap::new();
        for (key, value) in &study.attrs {
            if let AttrKey::User(k) = key {
                user_attrs.insert(k.to_string(), value.clone());
            }
        }
        Ok(user_attrs)
    }

    #[getter(_study_id)]
    pub fn id(&self) -> u32 {
        self.study.id
    }

    #[getter(study_name)]
    pub fn name(&self) -> &str {
        &self.study.name
    }

    #[getter(_storage)]
    pub fn storage<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.storage_pyobj.clone_ref(py)
    }

    #[getter]
    pub fn sampler<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sampler_pyobj.clone_ref(py)
    }

    #[getter]
    pub fn trial_queue<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.trial_queue_pyobj.clone_ref(py)
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
    pub fn best_trials(&self) -> PyResult<Vec<PyPersistedTrial>> {
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

fn resolve_storage(storage: Py<PyAny>) -> PyResult<(SharedStorage, Py<PyAny>)> {
    Python::attach(|py| {
        let storage_pyobj = storage.clone_ref(py);
        let storage_ref = storage.bind(py);
        if storage_ref.is_instance_of::<PyStorage>() {
            let storage = storage_ref.extract::<PyStorage>().map_err(|e| {
                PyValueError::new_err(format!("Failed to extract PyStorage: {e:?}"))
            })?;
            Ok::<_, PyErr>((storage.storage, storage_pyobj))
        } else {
            let mut wrapped = PyObjectStorage::new(storage);
            wrapped.sync_studies(true).map_err(err_to_exceptions)?;
            let wrapped: SharedStorage = Arc::new(RwLock::new(wrapped));
            Ok((wrapped, storage_pyobj))
        }
    })
}

#[pyfunction]
#[pyo3(name = "copy_study", signature = (*, from_study_name, from_storage, to_storage, to_study_name = None))]
pub fn py_copy_study(
    from_study_name: String,
    from_storage: Py<PyAny>,
    to_storage: Py<PyAny>,
    to_study_name: Option<String>,
) -> PyResult<()> {
    let (from_storage_arc, _) = resolve_storage(from_storage)?;
    let (from_directions, from_attrs, trials) = {
        let mut guard = from_storage_arc
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let from_study_id = guard
            .get_studies()
            .map_err(err_to_exceptions)?
            .iter()
            .find(|s| s.name == from_study_name)
            .map(|s| s.id)
            .ok_or(PyRuntimeError::new_err(format!(
                "Study {from_study_name} not found"
            )))?;
        let study = guard
            .get_study(from_study_id)
            .map_err(err_to_exceptions)?
            .clone();
        let trials = guard
            .get_trials(from_study_id)
            .map_err(err_to_exceptions)?
            .clone();
        (study.directions, study.attrs, trials)
    };

    let copied_study_name = to_study_name.unwrap_or(from_study_name);
    let (to_storage_arc, _) = resolve_storage(to_storage)?;
    let mut guard = to_storage_arc
        .write()
        .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
    let to_study_id = guard
        .create_new_study(&copied_study_name, from_directions)
        .map_err(err_to_exceptions)?
        .id;
    if !from_attrs.is_empty() {
        guard
            .set_study_attrs(to_study_id, from_attrs, false)
            .map_err(err_to_exceptions)?;
    }
    for trial in trials {
        guard
            .create_new_trial_from_template(to_study_id, &trial)
            .map_err(err_to_exceptions)?;
    }
    Ok(())
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
