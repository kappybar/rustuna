use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;
use pyo3::PyObject;
use pyo3::{prelude::*, types::PyType};

use rustuna_core::sampler::{Context as SamplerContext, RandomSampler, Sampler};
use rustuna_core::storage::Storage;
use rustuna_samplers::tpe::TpeSampler;

use crate::distribution::PyDistribution;
use crate::storage::PyStorage;
use crate::study::PyDirection;

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
    #[pyo3(signature = (seed = None))]
    fn tpe(_cls: &PyType, seed: Option<u64>) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => TpeSampler::seed_from_u64(seed),
            None => TpeSampler::new(),
        };
        Ok(PySampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
            kind: "tpe",
        })
    }

    #[classmethod]
    #[pyo3(signature = (seed = None))]
    fn random(_cls: &PyType, seed: Option<u64>) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => RandomSampler::seed_from_u64(seed),
            None => RandomSampler::new(),
        };
        Ok(PySampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
            kind: "random",
        })
    }

    #[classmethod]
    #[pyo3(signature = (seed = None, population_size = 50, mutation_prob = None, crossover_prob = 0.9, swapping_prob = 0.1))]
    fn nsgaii(
        _cls: &PyType,
        seed: Option<u64>,
        population_size: usize,
        mutation_prob: Option<f64>,
        crossover_prob: f64,
        swapping_prob: f64,
    ) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => rustuna_samplers::nsgaii::NSGAIISampler::seed_from_u64(
                seed,
                population_size,
                mutation_prob,
                crossover_prob,
                swapping_prob,
            ),
            None => rustuna_samplers::nsgaii::NSGAIISampler::new(
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
    fn support_joint_sampling(&self) -> bool {
        self.sampler.lock().unwrap().support_joint_sampling()
    }

    fn sample_independent(
        &self,
        ctx: &PySamplerContext,
        storage: &PyStorage,
        name: &str,
        distribution: &PyDistribution,
    ) -> PyResult<f64> {
        self.sampler
            .lock()
            .unwrap()
            .sample_independent(
                &ctx.context.clone(),
                storage.storage.clone(),
                name,
                &distribution.distribution,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample independent: {e:?}")))
    }

    fn sample_joint(
        &self,
        ctx: &PySamplerContext,
        storage: &PyStorage,
        search_space: HashMap<String, PyDistribution>,
    ) -> PyResult<HashMap<String, f64>> {
        self.sampler
            .lock()
            .unwrap()
            .sample_joint(
                &ctx.context.clone(),
                storage.storage.clone(),
                &search_space
                    .into_iter()
                    .map(|(k, v)| (k, v.distribution))
                    .collect(),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample joint: {e:?}")))
    }
}

#[derive(Clone)]
#[pyclass(name = "SamplerContext")]
#[pyo3(module = "rustuna")]
pub struct PySamplerContext {
    context: SamplerContext,
}
impl From<SamplerContext> for PySamplerContext {
    fn from(item: SamplerContext) -> Self {
        PySamplerContext { context: item }
    }
}
#[allow(non_local_definitions)]
#[pymethods]
impl PySamplerContext {
    #[new]
    #[pyo3(signature = (study_id, trial_number, directions))]
    pub fn py_new(
        study_id: u32,
        trial_number: u32,
        directions: Vec<PyDirection>,
    ) -> PyResult<Self> {
        Ok(PySamplerContext {
            context: SamplerContext {
                study_id,
                trial_number,
                directions: directions.into_iter().map(|d| d.into()).collect(),
            },
        })
    }
    #[getter]
    fn study_id(&self) -> u32 {
        self.context.study_id
    }
    #[getter]
    fn trial_number(&self) -> u32 {
        self.context.trial_number
    }
    #[getter]
    fn directions(&self) -> Vec<PyDirection> {
        self.context
            .directions
            .iter()
            .map(|d| d.clone().into())
            .collect()
    }
}

pub struct PyObjectSampler {
    obj: PyObject,
}
impl PyObjectSampler {
    pub fn new(obj: PyObject) -> Self {
        PyObjectSampler { obj }
    }
}
impl Sampler for PyObjectSampler {
    fn sample_independent(
        &mut self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn rustuna_core::storage::Storage>>,
        name: &str,
        distribution: &rustuna_core::distribution::Distribution,
    ) -> rustuna_core::Result<f64> {
        let mut guard = storage.write().unwrap();
        let study = guard.get_study(ctx.study_id)?;
        let study_attrs = study.attrs.clone();
        drop(guard);

        Python::with_gil(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = PyStorage {
                storage: storage.clone(),
                kind: "unset",
            };
            let py_distribution = PyDistribution::new(distribution.clone(), name, &study_attrs);
            let py_result = self
                .obj
                .call_method1(
                    py,
                    "sample_independent",
                    (py_ctx, py_storage, name, py_distribution),
                )
                .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::SamplerError))?;
            let py_result_ref = py_result.as_ref(py);
            let ret = py_result_ref
                .extract::<f64>()
                .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::SamplerError))?;
            Ok(ret)
        })
    }

    fn support_joint_sampling(&self) -> bool {
        Python::with_gil(|py| {
            self.obj
                .getattr(py, "support_joint_sampling")
                .map(|x| x.extract::<bool>(py).unwrap())
                .unwrap_or(false)
        })
    }

    fn sample_joint(
        &mut self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn Storage>>,
        search_space: &HashMap<String, rustuna_core::distribution::Distribution>,
    ) -> rustuna_core::Result<HashMap<String, f64>> {
        let mut guard = storage.write().unwrap();
        let study = guard.get_study(ctx.study_id)?;
        let study_attrs = study.attrs.clone();
        drop(guard);

        Python::with_gil(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = PyStorage {
                storage: storage.clone(),
                kind: "unset",
            };
            let py_search_space = PyDict::new(py);
            for (k, v) in search_space {
                let py_distribution = PyDistribution::new(v.clone(), k, &study_attrs).into_py(py);
                py_search_space
                    .set_item(k, py_distribution)
                    .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::SamplerError))?;
            }
            let py_result = self
                .obj
                .call_method1(py, "sample_joint", (py_ctx, py_storage, py_search_space))
                .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::SamplerError))?;
            let py_result = py_result.extract::<HashMap<String, f64>>(py).unwrap();
            Ok(py_result)
        })
    }
}
