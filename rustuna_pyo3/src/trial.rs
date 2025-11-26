use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustuna_core::attr::{AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;

use rustuna_core::trial::{PersistedTrial, Trial, TrialStateValues};

use crate::attrs::pyobj_to_attrs;
use crate::distribution::{
    category_label_to_pyobject, py_to_external_repr, pyobject_to_category_label, PyDistribution,
};

#[derive(Clone, Debug)]
#[pyclass(name = "TrialState")]
#[pyo3(module = "rustuna")]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTrialState {
    RUNNING = 0,
    COMPLETE = 1,
    PRUNED = 2,
    WAITING = 3,
    FAIL = 4,
}
impl From<TrialStateValues> for PyTrialState {
    fn from(item: TrialStateValues) -> Self {
        match item {
            TrialStateValues::Running => PyTrialState::RUNNING,
            TrialStateValues::Complete(_) => PyTrialState::COMPLETE,
            TrialStateValues::Pruned => PyTrialState::PRUNED,
            TrialStateValues::Fail => PyTrialState::FAIL,
            TrialStateValues::Waiting => PyTrialState::WAITING,
        }
    }
}
#[pymethods]
impl PyTrialState {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            PyTrialState::COMPLETE | PyTrialState::PRUNED | PyTrialState::FAIL
        )
    }
}

#[pyclass(name = "Trial")]
#[pyo3(module = "rustuna")]
pub struct PyTrial(Trial);
impl From<Trial> for PyTrial {
    fn from(item: Trial) -> Self {
        PyTrial(item)
    }
}
#[pymethods]
impl PyTrial {
    #[getter]
    pub fn study_id(&self) -> PyResult<u32> {
        Ok(self.0.study_id)
    }
    #[getter]
    pub fn number(&self) -> PyResult<u32> {
        Ok(self.0.number)
    }
    #[pyo3(signature = (name, low, high, step=None, log=false))]
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> PyResult<f64> {
        let dist = Distribution::Float {
            low,
            high,
            step,
            log,
        };
        let value = self.0.suggest(name, &dist).map_err(|e| match e.kind {
            rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                "The TPE sampler of rustuna currently only supports single objective study.",
            ),
            _ => PyRuntimeError::new_err(format!("Failed to suggest float: {:?}", e.kind)),
        })?;
        Ok(value)
    }
    #[pyo3(signature = (name, low, high, step=None, log=false))]
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: Option<i64>,
        log: bool,
    ) -> PyResult<i64> {
        let dist = Distribution::Int {
            low,
            high,
            step,
            log,
        };
        let value = self.0.suggest(name, &dist).map_err(|e| match e.kind {
            rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                "The TPE sampler of rustuna currently only supports single objective study.",
            ),
            _ => PyRuntimeError::new_err(format!("Failed to suggest int: {:?}", e.kind)),
        })?;
        Ok(value as i64)
    }
    #[pyo3(signature = (name, choices))]
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<PyObject>,
    ) -> PyResult<PyObject> {
        let mut category_labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
        let category_labels = Python::with_gil(|py| {
            for choice in choices {
                match pyobject_to_category_label(py, choice) {
                    Ok(label) => category_labels.push(label),
                    Err(e) => return Err(e),
                }
            }
            Ok(category_labels)
        })?;
        let label = self
            .0
            .suggest_categorical_enum(name, &category_labels)
            .map_err(|e| match e.kind {
                rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                    "The TPE sampler of rustuna currently only supports single objective study.",
                ),
                _ => {
                    PyRuntimeError::new_err(format!("Failed to suggest categorical: {:?}", e.kind))
                }
            })?;

        let return_value = Python::with_gil(|py| category_label_to_pyobject(py, label));
        Ok(return_value)
    }
    #[pyo3(signature = (key, value))]
    pub fn set_user_attr(&mut self, key: &str, value: String) -> PyResult<()> {
        self.0.set_user_attr(key, value).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to set user attr: {:?}", e.kind))
        })?;
        Ok(())
    }
}

#[derive(Debug)]
#[pyclass(name = "PersistedTrial")]
#[pyo3(module = "rustuna")]
pub struct PyPersistedTrial(PersistedTrial, Attrs);
impl PyPersistedTrial {
    pub fn new(trial: PersistedTrial, study_attrs: Attrs) -> Self {
        PyPersistedTrial(trial, study_attrs)
    }
}
#[allow(non_local_definitions)]
#[pymethods]
impl PyPersistedTrial {
    #[new]
    #[pyo3(signature = (study_id, number, state, values=None, internal_params=None, distributions=None, user_attrs=None, system_attrs=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        study_id: u32,
        number: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
        internal_params: Option<HashMap<String, f64>>,
        distributions: Option<HashMap<String, PyDistribution>>,
        user_attrs: Option<HashMap<String, String>>,
        system_attrs: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        if matches!(state, PyTrialState::COMPLETE) && values.is_none() {
            Err(PyValueError::new_err(
                "values must be specified when state is COMPLETE",
            ))?;
        }
        let mut trial = PersistedTrial::new(study_id, number);
        trial.state_values = match state {
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::COMPLETE => TrialStateValues::Complete(values.ok_or(
                PyValueError::new_err("values must be specified when state is COMPLETE"),
            )?),
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::FAIL => TrialStateValues::Fail,
            PyTrialState::WAITING => TrialStateValues::Waiting,
        };

        trial.internal_params = internal_params.unwrap_or_default();
        trial.distributions = HashMap::with_capacity(match &distributions {
            Some(d) => d.len(),
            None => 0,
        });
        for (name, dist) in distributions.unwrap_or_default() {
            trial.distributions.insert(name, dist.distribution);
        }

        let user_attrs = user_attrs.unwrap_or_default();
        let system_attrs = system_attrs.unwrap_or_default();
        let n_user_attrs = user_attrs.len();
        let n_system_attrs = user_attrs.len();
        let mut trial_attrs = Attrs::with_capacity(n_user_attrs + n_system_attrs);
        for (key, value) in user_attrs {
            trial_attrs.insert(AttrKey::User(key), value);
        }
        for (key, value) in system_attrs {
            trial_attrs.insert(AttrKey::System(key), value);
        }
        trial.attrs = trial_attrs;

        let study_attrs = Attrs::new();
        Ok(PyPersistedTrial(trial, study_attrs))
    }

    #[getter]
    fn study_id(&self) -> PyResult<u32> {
        Ok(self.0.study_id)
    }

    #[getter]
    fn number(&self) -> PyResult<u32> {
        Ok(self.0.number)
    }

    #[getter]
    fn state(&self) -> PyResult<PyTrialState> {
        Ok(PyTrialState::from(self.0.state_values.clone()))
    }

    #[getter]
    fn values(&self) -> Option<Vec<f64>> {
        match self.0.state_values {
            TrialStateValues::Complete(ref values) => Some(values.clone()),
            _ => None,
        }
    }

    #[getter]
    fn distributions(&self) -> PyResult<HashMap<String, PyDistribution>> {
        let mut distributions = HashMap::new();
        for (name, dist) in &self.0.distributions {
            let distribution = PyDistribution::new(dist.clone(), name, &self.1);
            distributions.insert(name.clone(), distribution);
        }
        Ok(distributions)
    }

    #[getter]
    fn internal_params(&self) -> PyResult<HashMap<String, f64>> {
        Ok(self.0.internal_params.clone())
    }

    #[getter]
    fn params(&self) -> PyResult<HashMap<String, PyObject>> {
        self.0
            .internal_params
            .iter()
            .map(|(name, internal_repr)| {
                let maybe_pyobj: PyResult<PyObject> = match self.0.distributions.get(name) {
                    Some(dist) => py_to_external_repr(dist, *internal_repr, name, &self.1),
                    None => Err(PyValueError::new_err(format!("No distribution for {name}"))),
                };
                maybe_pyobj.map(|v| (name.to_string(), v))
            })
            .collect()
    }

    #[getter]
    fn user_attrs(&self) -> PyResult<HashMap<String, String>> {
        let user_attrs = self
            .0
            .attrs
            .iter()
            .filter_map(|(key, value)| match key {
                AttrKey::User(k) => Some((k.clone(), value.clone())),
                _ => None,
            })
            .collect();
        Ok(user_attrs)
    }

    #[getter]
    fn system_attrs(&self) -> PyResult<HashMap<String, String>> {
        let system_attrs = self
            .0
            .attrs
            .iter()
            .filter_map(|(key, value)| match key {
                AttrKey::System(k) => Some((k.clone(), value.clone())),
                _ => None,
            })
            .collect();
        Ok(system_attrs)
    }

    fn __repr__(slf: &PyCell<Self>) -> PyResult<String> {
        let class_name: &str = slf.get_type().name()?;
        Ok(format!("{}({})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        let params: PyResult<String> =
            Python::with_gil(|py| Ok(self.params()?.to_object(py).to_string()));
        Ok(format!(
            "number={} state={:?} values={:?} params={} distributions={:?} user_attrs={:?} system_attrs={:?}",
            self.number()?,
            self.state()?,
            self.values(),
            params?,
            self.distributions()?,
            self.user_attrs()?,
            self.system_attrs()?,
        ))
    }
}

pub fn pyobject_to_persisted_trial(trial: &PyAny, study_id: u32) -> PyResult<PersistedTrial> {
    let number = trial.getattr("number")?.extract::<u32>()?;
    let mut persisted_trial = PersistedTrial::new(study_id, number);

    let state = trial.getattr("state")?.extract::<PyTrialState>()?;
    let values = trial.getattr("values")?.extract::<Option<Vec<f64>>>()?;
    persisted_trial.state_values = match state {
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

    let src_internal_params = trial.getattr("internal_params")?;
    if !src_internal_params.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("internal_params must be a dict"));
    }
    let src_internal_params = src_internal_params.downcast::<PyDict>()?;
    let mut internal_params: HashMap<String, f64> =
        HashMap::with_capacity(src_internal_params.len());
    for (key, value) in src_internal_params.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<f64>()?;
        internal_params.insert(key, value);
    }
    persisted_trial.internal_params = internal_params;

    let src_distributions = trial.getattr("distributions")?;
    if !src_distributions.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("distributions must be a dict"));
    }
    let src_distributions = src_distributions.downcast::<PyDict>()?;
    let mut distributions: HashMap<String, Distribution> =
        HashMap::with_capacity(src_distributions.len());
    for (key, value) in src_distributions.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<PyDistribution>()?;
        distributions.insert(key, value.into());
    }
    persisted_trial.distributions = distributions;

    let user_attrs = trial.getattr("user_attrs")?;
    let system_attrs = trial.getattr("system_attrs")?;
    if !user_attrs.is_instance_of::<PyDict>() || !system_attrs.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(
            "user_attrs and system_attrs must be a dict",
        ));
    }
    persisted_trial.attrs = pyobj_to_attrs(user_attrs, system_attrs)?;
    Ok(persisted_trial)
}
