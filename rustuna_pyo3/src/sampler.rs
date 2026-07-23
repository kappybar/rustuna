use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::Py;
use pyo3::{prelude::*, types::PyType};

use rustuna_core::sampler::Sampler;
use rustuna_core::storage::Storage;
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::tpe::{TpeConfig, TpeSampler};

pub mod cmaes;
mod context;
pub mod python;
pub mod random;
pub use context::PySamplerContext;

use crate::distribution::PyDistribution;
use crate::pyobject_storage::PyPyObjectStorage;
use crate::storage::PyStorage;
use crate::trial::PyTrialState;

#[derive(Clone)]
#[pyclass(name = "Sampler")]
#[pyo3(module = "rustuna")]
pub struct PySampler {
    pub sampler: Arc<Mutex<dyn Sampler>>,
    pub kind: &'static str,
}
#[pymethods]
impl PySampler {
    #[classmethod]
    #[pyo3(signature = (seed = None, n_startup_trials = 10, multivariate = true))]
    fn tpe(
        _cls: &Bound<'_, PyType>,
        seed: Option<u64>,
        n_startup_trials: usize,
        multivariate: bool,
    ) -> PyResult<Self> {
        let rs_sampler = TpeSampler::from_config(TpeConfig {
            seed,
            n_startup_trials,
            multivariate,
        });
        Ok(PySampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
            kind: "tpe",
        })
    }

    #[classmethod]
    #[pyo3(signature = (seed = None, population_size = 50, mutation_prob = None, crossover_prob = 0.9, swapping_prob = 0.1))]
    fn nsgaii(
        _cls: &Bound<'_, PyType>,
        seed: Option<u64>,
        population_size: usize,
        mutation_prob: Option<f64>,
        crossover_prob: f64,
        swapping_prob: f64,
    ) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => rustuna_sampler::nsgaii::NSGAIISampler::seed_from_u64(
                seed,
                population_size,
                mutation_prob,
                crossover_prob,
                swapping_prob,
            ),
            None => rustuna_sampler::nsgaii::NSGAIISampler::new(
                population_size,
                mutation_prob,
                crossover_prob,
                swapping_prob,
            ),
        };
        Ok(PySampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
            kind: "nsgaii",
        })
    }

    #[getter]
    fn support_joint_sampling(&self) -> PyResult<bool> {
        let guard = self
            .sampler
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire sampler lock"))?;
        Ok(guard.support_joint_sampling())
    }

    fn sample_independent(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        name: &str,
        distribution: &PyDistribution,
    ) -> PyResult<f64> {
        let arc_storage = extract_storage(storage)?;
        self.sampler
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the sampler guard"))?
            .sample_independent(
                &ctx.context.clone(),
                arc_storage,
                name,
                &distribution.distribution,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample independent: {e:?}")))
    }

    fn sample_joint(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        search_space: HashMap<String, PyDistribution>,
    ) -> PyResult<HashMap<String, f64>> {
        let arc_storage = extract_storage(storage)?;
        self.sampler
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the sampler guard"))?
            .sample_joint(
                &ctx.context.clone(),
                arc_storage,
                &search_space
                    .into_iter()
                    .map(|(k, v)| (k, v.distribution))
                    .collect(),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample joint: {e:?}")))
    }

    #[pyo3(signature = (ctx, storage, state, values = None))]
    fn after_trial(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let arc_storage = extract_storage(storage)?;
        let state_values = match state {
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::COMPLETE => TrialStateValues::Complete(values.ok_or(
                PyRuntimeError::new_err("values must be specified when state is COMPLETE"),
            )?),
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::WAITING => TrialStateValues::Waiting,
            PyTrialState::FAIL => TrialStateValues::Fail,
        };
        self.sampler
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the sampler guard"))?
            .after_trial(&ctx.context, arc_storage, &state_values)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to call after_trial: {e:?}")))
    }
}

fn extract_storage(storage: Py<PyAny>) -> PyResult<Arc<RwLock<dyn Storage>>> {
    Python::attach(|py| {
        let storage_ref = storage.bind(py);
        if let Ok(py_storage) = storage_ref.extract::<PyStorage>() {
            Ok(py_storage.storage.clone())
        } else if let Ok(py_obj_storage) = storage_ref.extract::<PyPyObjectStorage>() {
            Ok(py_obj_storage.storage.clone() as Arc<RwLock<dyn Storage>>)
        } else {
            Err(PyRuntimeError::new_err(
                "Invalid storage type. Use rustuna.Storage for Rust-native storages or rustuna.PyObjectStorage for Python StorageProtocol implementations.",
            ))
        }
    })
}
