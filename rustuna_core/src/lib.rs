pub use error::{Error, ErrorKind};

pub mod attr;
pub mod distribution;
pub mod sampler;
pub mod storage;
pub mod study;
pub mod trial;

mod error;
mod study_cache;

/// This is a custom `Result` type for this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
