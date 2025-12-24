use std::collections::HashMap;

use chrono::NaiveDateTime;
use rustuna_core::trial::PersistedTrial;
use rustuna_core::Result;
use serde::{Deserialize, Serialize};

pub trait OptunaCompatibleStorage: Send + Sync {
    fn get_study_id_trial_number_from_trial_id(&mut self, trial_id: u32) -> Result<(u32, u32)>;
    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> Result<u32>;
    fn set_trial_datetime(
        &mut self,
        trial_id: u32,
        datetime_start: Option<NaiveDateTime>,
        datetime_complete: Option<NaiveDateTime>,
    ) -> Result<()>;
    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()>;
    fn get_trials_diff_optuna(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
    ) -> Result<Vec<PersistedTrial>>;
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
