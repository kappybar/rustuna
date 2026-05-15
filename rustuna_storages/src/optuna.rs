use std::collections::HashMap;

use rustuna_core::Result;
use serde::{Deserialize, Serialize};

/// Optional extension trait for storage backends that preserve Optuna compatibility data.
///
/// Rustuna itself stores trial intermediate values and attributes as strings, but converters that
/// interoperate with Optuna sometimes need extra structured metadata. Backends implementing this
/// trait can persist such compatibility-specific state directly.
pub trait OptunaCompatibleStorage: Send + Sync {
    /// Stores all intermediate values for a trial in a backend-specific format.
    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()>;
}

/// Intermediate value entry for JSON serialization.
///
/// This structure is used to serialize intermediate values with their type information,
/// preserving special float values (NaN, Infinity, -Infinity) that cannot be represented
/// in standard JSON format.
///
/// # Fields
/// * `step` - The step number (epoch, iteration, etc.) for this intermediate value
/// * `value` - The actual f64 value. None for special values (NaN, Infinity, -Infinity)
/// * `value_type` - Type discriminator. One of:
///   - "FINITE": Normal floating-point value (value is Some)
///   - "NAN": Not a Number (value is None)
///   - "INF_POS": Positive infinity (value is None)
///   - "INF_NEG": Negative infinity (value is None)
#[derive(Debug, Serialize, Deserialize)]
pub struct IntermediateValueEntry {
    pub step: u32,
    pub value: Option<f64>,
    pub value_type: String,
}
