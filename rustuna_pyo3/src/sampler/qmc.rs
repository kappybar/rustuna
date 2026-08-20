use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Py;

use rustuna_core::sampler::Sampler;
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::qmc::QmcSampler;

use crate::distribution::PyDistribution;
use crate::sampler::{extract_storage, PySamplerContext};
use crate::trial::PyTrialState;

/// A quasi-Monte Carlo sampler backed by the Sobol' sequence.
///
/// The sampler covers the search space with a low-discrepancy sequence instead of independent
/// random draws, so a given number of trials spreads more evenly than random search does. The
/// sequence matches `scipy.stats.qmc.Sobol(d, scramble=False)`, which is what Optuna's
/// `QMCSampler` uses under its default settings. Scrambling is not implemented yet.
///
/// Sobol' points are a quadrature rule, so they are most uniform when the number of trials is a
/// power of two. Parameters outside the joint search space, including everything in the first
/// trial, fall back to random sampling.
///
/// The position in the sequence is kept in a study system attribute rather than in the sampler,
/// so workers sharing a storage walk one sequence together and a resumed study continues where it
/// left off. Threads within one process are serialized by the storage lock, but two processes can
/// reserve the same index, because reading and writing the counter is not a single storage
/// transaction.
#[derive(Clone)]
#[pyclass(name = "QMCSampler", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyQmcSampler {
    pub sampler: Arc<Mutex<QmcSampler>>,
}
#[pymethods]
impl PyQmcSampler {
    /// Creates a sampler.
    ///
    /// Args:
    ///     seed: Seed of the random sampler used for parameters outside the joint search space.
    ///         The Sobol' sequence itself is deterministic and ignores this.
    #[new]
    #[pyo3(signature = (*, seed = None))]
    fn py_new(seed: Option<u64>) -> Self {
        let rs_sampler = match seed {
            Some(seed) => QmcSampler::seed_from_u64(seed),
            None => QmcSampler::new(),
        };
        PyQmcSampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
        }
    }

    #[getter]
    fn support_joint_sampling(&self) -> PyResult<bool> {
        Ok(true)
    }

    fn sample_independent(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        name: &str,
        distribution: &PyDistribution,
    ) -> PyResult<f64> {
        let arc_storage = extract_storage(storage)?;
        let context = ctx.context.clone();
        let name = name.to_owned();
        let distribution = distribution.distribution.clone();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .sample_independent(&context, arc_storage, &name, &distribution)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to sample independent: {e:?}"))
                })
        })
    }

    fn sample_joint(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        search_space: HashMap<String, PyDistribution>,
    ) -> PyResult<HashMap<String, f64>> {
        let arc_storage = extract_storage(storage)?;
        let context = ctx.context.clone();
        let search_space = search_space
            .into_iter()
            .map(|(k, v)| (k, v.distribution))
            .collect();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .sample_joint(&context, arc_storage, &search_space)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample joint: {e:?}")))
        })
    }

    #[pyo3(signature = (ctx, storage, state, values = None))]
    fn after_trial(
        &self,
        py: Python<'_>,
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
        let context = ctx.context.clone();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .after_trial(&context, arc_storage, &state_values)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to call after_trial: {e:?}")))
        })
    }
}
