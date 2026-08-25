//! Sobol' low-discrepancy sequence generator.
//!
//! This module reproduces `scipy.stats.qmc.Sobol(d, scramble=False)` exactly: it uses the same
//! Joe-Kuo direction numbers, the same Antonov-Saleev Gray code construction, and the same
//! convention of placing the origin at index 0.
//!
//! A coordinate is the exclusive-or of the direction numbers selected by the set bits of the
//! Gray code of the point index. Since every operation is a bitwise exclusive-or, the direction
//! numbers are kept as fixed-point integers and scaled into `[0, 1)` only when a point is handed
//! out.
//!
//! # Balance properties
//!
//! Sobol' points are a quadrature rule, not a stream of independent samples. They form a
//! `(t, m, d)`-net only when the `2^m` points at indices `0..2^m` are taken together, so using a
//! non-power-of-two number of points, skipping index 0, or thinning the sequence all degrade
//! their uniformity.
//!
//! # Examples
//!
//! ```
//! use rustuna_core::Result;
//! use rustuna_sampler::qmc::sobol;
//!
//! fn main() -> Result<()> {
//!     assert_eq!(sobol::nth_point(2, 0)?, vec![0.0, 0.0]);
//!     assert_eq!(sobol::nth_point(2, 1)?, vec![0.5, 0.5]);
//!     assert_eq!(sobol::nth_point(2, 2)?, vec![0.75, 0.25]);
//!     assert_eq!(sobol::nth_point(2, 3)?, vec![0.25, 0.75]);
//!     Ok(())
//! }
//! ```

pub mod direction_numbers;
mod engine;

pub use engine::nth_point;
