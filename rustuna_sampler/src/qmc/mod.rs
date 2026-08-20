//! Quasi-Monte Carlo sampling based on the Sobol' sequence.

pub mod direction_numbers;
mod sampler;
mod sobol;

pub use sampler::QmcSampler;
pub use sobol::SobolEngine;
