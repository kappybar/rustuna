//! Core components for Rustuna.
//!
//! This crate provides the central study, trial, storage, sampler, queue, and distribution
//! components used across Rustuna. Concrete storage backends, sampler implementations, and
//! language bindings are implemented in other crates in the workspace.

pub use error::{Error, ErrorKind};

pub mod attr;
pub mod datetime;
pub mod distribution;
pub mod multi_objective;
pub mod parzen_estimator;
pub mod sampler;
pub mod storage;
pub mod string_interner;
pub mod study;
pub mod study_cache;
pub mod transform;
pub mod trial;
pub mod trial_queue;

mod error;
mod parzen_estimator;

/// Implementation details shared by Rustuna crates.
///
/// This module is not part of Rustuna's stable public API. Items in this module
/// are not covered by Rustuna's semantic-versioning guarantees and may be
/// changed or removed in any release without a major version bump.
#[doc(hidden)]
pub mod internal {
    pub mod parzen_estimator {
        pub use crate::parzen_estimator::ParzenEstimator;
    }
}

/// A crate-specific [`std::result::Result`] alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
