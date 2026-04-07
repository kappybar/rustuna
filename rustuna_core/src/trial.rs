use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::attr::{category_labels_to_attrs, AttrKey, Attrs, CategoryLabel};
use crate::distribution::Distribution;
use crate::sampler::{Context as SamplerContext, Sampler};
use crate::storage::Storage;
use crate::study::Direction;
use crate::{Error, ErrorKind, Result};

/// Trial is a struct that is equivalent to `optuna.trial.Trial`.
pub struct Trial {
    pub id: u32,
    pub study_id: u32,
    pub number: u32,
    pub datetime_start: Option<String>,
    pub datetime_complete: Option<String>,
    directions: Vec<Direction>,
    storage: Arc<RwLock<dyn Storage>>,
    sampler: Arc<Mutex<dyn Sampler>>,
    joint_params: HashMap<String, (Distribution, f64)>,
    fixed_params: HashMap<String, CategoryLabel>,
    cached_trial: PersistedTrial,
}
impl Trial {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trial_id: u32,
        study_id: u32,
        number: u32,
        datetime_start: Option<String>,
        datetime_complete: Option<String>,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
        sampler: Arc<Mutex<dyn Sampler>>,
        joint_params: HashMap<String, (Distribution, f64)>,
        fixed_params: HashMap<String, CategoryLabel>,
    ) -> Self {
        let mut cached_trial = PersistedTrial::new(trial_id, study_id, number);
        cached_trial.datetime_start = datetime_start.clone();
        cached_trial.datetime_complete = datetime_complete.clone();
        Trial {
            id: trial_id,
            study_id,
            number,
            datetime_start,
            datetime_complete,
            directions,
            storage,
            sampler,
            joint_params,
            fixed_params,
            cached_trial,
        }
    }

    pub fn suggest(&mut self, name: &str, distribution: &Distribution) -> Result<f64> {
        if let Some(fixed_value) = self.fixed_params.get(name) {
            let result = if let Distribution::Categorical { cardinality } = distribution {
                // CategoricalDistribution
                let mut storage_guard = self
                    .storage
                    .write()
                    .map_err(|_| Error::new(ErrorKind::StorageError))?;
                let labels =
                    storage_guard.get_category_labels(self.study_id, name, *cardinality)?;
                drop(storage_guard);

                labels.and_then(|labels| {
                    labels
                        .iter()
                        .position(|l| l == fixed_value)
                        .map(|index| index as f64)
                })
            } else {
                // IntDistribution and FloatDistribution
                // Note: step and log validation are intentionally omitted, matching Optuna's behavior.
                match (fixed_value, distribution) {
                    (CategoryLabel::Float(f), Distribution::Float { .. }) => Some(*f),
                    (CategoryLabel::Int(i), Distribution::Float { .. }) => Some(*i as f64),
                    (CategoryLabel::Int(i), Distribution::Int { .. }) => Some(*i as f64),
                    (CategoryLabel::Float(f), Distribution::Int { .. }) => Some(*f as i64 as f64),
                    _ => None,
                }
                .filter(|&v| distribution.contains(v))
            };

            if let Some(internal_value) = result {
                let mut storage_guard = self
                    .storage
                    .write()
                    .map_err(|_| Error::new(ErrorKind::StorageError))?;
                storage_guard.set_trial_param(self.id, name, distribution, internal_value)?;
                drop(storage_guard);

                self.cached_trial
                    .internal_params
                    .insert(name.to_string(), internal_value);
                self.cached_trial
                    .distributions
                    .insert(name.to_string(), distribution.clone());

                return Ok(internal_value);
            }
        }

        if let Some((d, val)) = self.joint_params.get(name) {
            if *d == *distribution {
                let param_value = *val;
                let mut storage_guard = self
                    .storage
                    .write()
                    .map_err(|_| Error::new(ErrorKind::StorageError))?;
                storage_guard.set_trial_param(self.id, name, distribution, param_value)?;
                drop(storage_guard);
                return Ok(param_value);
            }
        }

        let context = SamplerContext {
            study_id: self.study_id,
            trial_number: self.number,
            trial_id: self.id,
            directions: self.directions.clone(),
        };
        let mut sampler_guard = self
            .sampler
            .lock()
            .map_err(|_| Error::new(ErrorKind::SamplerError))?;
        let param_value =
            sampler_guard.sample_independent(&context, self.storage.clone(), name, distribution)?;
        drop(sampler_guard);

        let mut storage_guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        storage_guard.set_trial_param(self.id, name, distribution, param_value)?;
        drop(storage_guard);

        self.cached_trial
            .internal_params
            .insert(name.to_string(), param_value);
        self.cached_trial
            .distributions
            .insert(name.to_string(), distribution.clone());

        Ok(param_value)
    }

    // Design Note:
    // suggest_float and suggest_int do not support `step` and `log` arguments to keep the API easy to use.
    // Users who want to use them need to call `suggest()` function directly.
    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        let distribution = Distribution::Float {
            low,
            high,
            step: None,
            log: false,
        };
        let param_value = self.suggest(name, &distribution)?;
        Ok(param_value)
    }

    pub fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        let distribution = Distribution::Int {
            low,
            high,
            step: 1,
            log: false,
        };
        let param_value = self.suggest(name, &distribution)?;
        Ok(param_value as i64)
    }

    pub fn suggest_categorical_enum<'a>(
        &'a mut self,
        name: &str,
        choices: &'a [CategoryLabel],
    ) -> Result<&'a CategoryLabel> {
        let study_id = self.study_id;
        let storage = self.storage.clone();

        // Save labels in the study system attr before calling suggest(),
        // so that fixed_params can look up category labels during suggest().
        let category_labels = category_labels_to_attrs(name, choices);
        let mut guard = storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        guard.set_study_attrs(study_id, category_labels, false)?;
        drop(guard);

        let c = self.suggest_categorical(name, choices)?;
        Ok(c)
    }

    pub fn suggest_categorical<'a, T>(&'a mut self, name: &str, choices: &'a [T]) -> Result<&'a T> {
        let distribution = Distribution::Categorical {
            cardinality: choices.len(),
        };
        let param_value = self.suggest(name, &distribution)?;
        if choices.len() <= (param_value as usize) {
            return Err(Error::new(ErrorKind::Unexpected));
        }
        let choice = &choices[param_value as usize];
        Ok(choice)
    }

    // Design Note:
    // Note that `AttrKind::System` should not be exposed to users,
    // as it may be updated without going through the `Trial` object,
    // making it difficult to cache the value.
    pub fn get_user_attr(&mut self, key: &str) -> Option<&String> {
        let key = AttrKey::User(key.into());
        self.cached_trial.attrs.get(&key)
    }

    pub fn set_user_attr(&mut self, key: &str, value: String) -> Result<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let mut attrs = Attrs::new();

        let key = AttrKey::User(key.into());
        attrs.insert(key.clone(), value.clone());
        guard.set_trial_attrs(self.id, attrs, false)?;
        self.cached_trial.attrs.insert(key, value);
        Ok(())
    }
}

/// PersistedTrial is a struct that represents a trial that has been persisted to storage.
/// This is equivalent to `optuna.trial.FrozenTrial`.
#[derive(Clone, Debug)]
pub struct PersistedTrial {
    pub id: u32,
    pub study_id: u32,
    pub number: u32,
    pub state_values: TrialStateValues,
    pub internal_params: HashMap<String, f64>,
    pub distributions: HashMap<String, Distribution>,
    pub attrs: Attrs,
    pub datetime_start: Option<String>,
    pub datetime_complete: Option<String>,
}
impl PersistedTrial {
    pub fn new(id: u32, study_id: u32, number: u32) -> PersistedTrial {
        PersistedTrial {
            id,
            study_id,
            number,
            state_values: TrialStateValues::Running,
            internal_params: HashMap::new(),
            distributions: HashMap::new(),
            attrs: Attrs::new(),
            datetime_start: None,
            datetime_complete: None,
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.state_values,
            TrialStateValues::Complete(_) | TrialStateValues::Pruned | TrialStateValues::Fail
        )
    }

    pub fn validate(&self) -> Result<()> {
        // TODO(c-bata): Consider introducing ErrorKind::TrialInvalid.
        if !matches!(self.state_values, TrialStateValues::Waiting) && self.datetime_start.is_none()
        {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                "datetime_start is supposed to be set when the trial state is not waiting."
                    .to_string(),
            ));
        }
        if self.is_finished() && self.datetime_complete.is_none() {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                "datetime_complete is supposed to be set for a finished trial.".to_string(),
            ));
        }
        if !self.is_finished() && self.datetime_complete.is_some() {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                "datetime_complete is supposed to be None for an unfinished trial.".to_string(),
            ));
        }
        if let TrialStateValues::Complete(values) = &self.state_values {
            if values.iter().any(|v| v.is_nan()) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "values should not contain NaN.".to_string(),
                ));
            }
        }
        if self.internal_params.len() != self.distributions.len() {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                format!(
                    "The number of parameters {} and distributions {} don't match.",
                    self.internal_params.len(),
                    self.distributions.len()
                ),
            ));
        }
        for (param_name, &internal_value) in &self.internal_params {
            let distribution = self.distributions.get(param_name).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Parameter '{param_name}' is not found in distributions."),
                )
            })?;
            if !distribution.contains(internal_value) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!(
                        "The value {internal_value} of parameter '{param_name}' isn't contained in the distribution {distribution:?}."
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum TrialStateValues {
    Running,
    Pruned,
    Complete(Vec<f64>),
    Fail,
    Waiting,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::sampler::RandomSampler;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study_with_arc;

    #[test]
    fn test_enqueue_and_suggest_float() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask(sampler.clone())?;
        let value = trial.suggest_float("x", 0.0, 10.0)?;
        assert_eq!(value, 5.0);
        Ok(())
    }

    #[test]
    fn test_enqueue_and_suggest_int() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Int(7));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask(sampler.clone())?;
        let value = trial.suggest_int("x", 0, 10)?;
        assert_eq!(value, 7);
        Ok(())
    }

    #[test]
    fn test_enqueue_fallback_on_out_of_range() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(100.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask(sampler.clone())?;
        let value = trial.suggest_float("x", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&value));
        Ok(())
    }

    #[test]
    fn test_enqueue_mixed_with_normal_ask() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        // First ask should use enqueued trial
        let mut trial1 = study.ask(sampler.clone())?;
        assert_eq!(trial1.suggest_float("x", 0.0, 10.0)?, 5.0);

        // Second ask should create a new trial (sampled)
        let mut trial2 = study.ask(sampler.clone())?;
        let value = trial2.suggest_float("x", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&value));

        Ok(())
    }

    #[test]
    fn test_enqueue_unspecified_param_sampled() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask(sampler.clone())?;
        assert_eq!(trial.suggest_float("x", 0.0, 10.0)?, 5.0);
        // "y" is not in fixed_params, so it should be sampled
        let y = trial.suggest_float("y", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&y));

        Ok(())
    }

    #[test]
    fn test_trial_user_attr() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), directions)?;

        // Set user attributes
        let mut trial = study.ask(sampler.clone())?;
        trial.set_user_attr("key", "user".to_string())?;

        // Set system attributes
        let mut guard = storage
            .write()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let mut attrs = Attrs::new();
        attrs.insert(AttrKey::System("key".into()), "system".to_string());
        guard.set_trial_attrs(trial.id, attrs, false)?;

        // Check the attributes
        let trial = guard.get_trial(trial.id)?;
        assert_eq!(
            trial.attrs.get(&AttrKey::User("key".into())),
            Some(&"user".to_string())
        );
        assert_eq!(
            trial.attrs.get(&AttrKey::System("key".into())),
            Some(&"system".to_string())
        );
        Ok(())
    }
}
