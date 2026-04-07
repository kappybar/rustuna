use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::attr::{extract_fixed_params, fixed_params_to_attrs, AttrKey, Attrs, CategoryLabel};
use crate::distribution::Distribution;
use crate::sampler::{Context as SamplerContext, Sampler};
use crate::storage::Storage;
use crate::trial::{PersistedTrial, Trial, TrialStateValues};
use crate::trial_queue::{InMemoryTrialQueue, TrialQueue};
use crate::{Error, ErrorKind, Result};

pub fn create_study<S: Storage + Send + Sync + 'static>(
    study_name: &str,
    mut storage: S,
    directions: Vec<Direction>,
) -> Result<Study> {
    let study_id = storage.create_new_study(study_name, directions.clone())?.id;
    let storage = Arc::new(RwLock::new(storage));
    Ok(Study::new(
        study_id,
        study_name.to_string(),
        directions,
        storage,
    ))
}

pub fn create_study_with_arc(
    study_name: &str,
    storage: Arc<RwLock<dyn Storage>>,
    directions: Vec<Direction>,
) -> Result<Study> {
    let mut guard = storage
        .write()
        .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
    let study_id = guard.create_new_study(study_name, directions.clone())?.id;
    drop(guard);
    Ok(Study::new(
        study_id,
        study_name.to_string(),
        directions,
        storage,
    ))
}

#[derive(Clone)]
pub struct Study {
    pub id: u32,
    pub name: String,
    pub directions: Vec<Direction>,
    pub storage: Arc<RwLock<dyn Storage>>,
    pub queue: Arc<RwLock<dyn TrialQueue>>,
}
impl Study {
    pub fn new(
        id: u32,
        name: String,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
    ) -> Self {
        let queue = Arc::new(RwLock::new(InMemoryTrialQueue::new()));
        Study {
            id,
            name,
            directions,
            storage,
            queue,
        }
    }

    pub fn with_queue(
        id: u32,
        name: String,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
        queue: Arc<RwLock<dyn TrialQueue>>,
    ) -> Self {
        Study {
            id,
            name,
            directions,
            storage,
            queue,
        }
    }

    pub fn from_id(id: u32, storage: Arc<RwLock<dyn Storage>>) -> Result<Self> {
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let study = guard.get_study(id)?;
        let name = study.name.clone();
        let directions = study.directions.clone();
        drop(guard);
        Ok(Study::new(id, name, directions, storage))
    }

    pub fn from_name(name: String, storage: Arc<RwLock<dyn Storage>>) -> Result<Self> {
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let studies = guard.get_studies()?;
        let study = studies
            .iter()
            .find(|s| s.name == name)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        let study_id = study.id;
        let directions = study.directions.clone();
        drop(guard);
        Ok(Study::new(study_id, name, directions, storage))
    }

    pub fn ask(&self, sampler: Arc<Mutex<dyn Sampler>>) -> Result<Trial> {
        let queued_trial_id = {
            let mut queue_guard = self.queue.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire a queue guard: {e}"),
                )
            })?;
            queue_guard.pop().ok()
        };

        let (trial_id, trial_number, datetime_start, datetime_complete, fixed_params) =
            if let Some(trial_id) = queued_trial_id {
                // Try to get trial from storage and transition to Running state.
                // If any storage operation fails, push the trial_id back to the queue.
                let result = (|| {
                    let mut guard = self.storage.write().map_err(|e| {
                        Error::with_reason(
                            ErrorKind::Unexpected,
                            format!("Failed to acquire a storage guard: {e}"),
                        )
                    })?;
                    let trial = guard.get_trial(trial_id)?;

                    let trial_number = trial.number;
                    let datetime_start = trial.datetime_start.clone();
                    let datetime_complete = trial.datetime_complete.clone();
                    let fixed_params = extract_fixed_params(&trial.attrs);

                    guard.set_trial_state_values(trial_id, TrialStateValues::Running)?;

                    Ok((
                        trial_id,
                        trial_number,
                        datetime_start,
                        datetime_complete,
                        fixed_params,
                    ))
                })();

                match result {
                    Ok(values) => values,
                    Err(e) => {
                        // Push the trial_id back to the queue on storage error
                        let mut queue_guard = self.queue.write().map_err(|queue_err| {
                            Error::with_reason(
                                ErrorKind::Unexpected,
                                format!("Failed to acquire queue guard for recovery: {queue_err}"),
                            )
                        })?;
                        let _ = queue_guard.push(trial_id);
                        return Err(e);
                    }
                }
            } else {
                let mut guard = self.storage.write().map_err(|e| {
                    Error::with_reason(
                        ErrorKind::Unexpected,
                        format!("Failed to acquire a storage guard: {e}"),
                    )
                })?;
                let trial = guard.create_new_trial(self.id)?;
                (
                    trial.id,
                    trial.number,
                    trial.datetime_start.clone(),
                    trial.datetime_complete.clone(),
                    HashMap::new(),
                )
            };

        let mut guard = sampler
            .lock()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let joint_params: HashMap<String, (Distribution, f64)> = if guard.support_joint_sampling() {
            let joint_search_space = self
                .storage
                .write()
                .map_err(|_| Error::new(ErrorKind::Unexpected))?
                .get_joint_search_space(self.id)?;

            let ctx = SamplerContext {
                study_id: self.id,
                trial_number,
                trial_id,
                directions: self.directions.clone(),
            };
            let params = guard.sample_joint(&ctx, self.storage.clone(), &joint_search_space)?;
            let mut joint_params = HashMap::new();
            for (name, param_value) in params {
                if !joint_search_space.contains_key(&name) {
                    continue;
                }
                let distribution = joint_search_space[&name].clone();
                joint_params.insert(name, (distribution, param_value));
            }
            joint_params
        } else {
            HashMap::new()
        };

        let trial = Trial::new(
            trial_id,
            self.id,
            trial_number,
            datetime_start,
            datetime_complete,
            self.directions.clone(),
            Arc::clone(&self.storage),
            sampler.clone(),
            joint_params,
            fixed_params,
        );
        Ok(trial)
    }

    pub fn tell(&self, trial_number: u32, state_values: TrialStateValues) -> Result<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let trial_id = guard.get_trial_id_from_study_id_trial_number(self.id, trial_number)?;
        guard.set_trial_state_values(trial_id, state_values)?;
        Ok(())
    }

    pub fn optimize<F>(
        &self,
        mut objective: F,
        // TODO(c-bata): Avoid to wrap Sampler by Arc and Mutex.
        // Sampler does not need to be shared across threads.
        sampler: Arc<Mutex<dyn Sampler>>,
        n_trials: usize,
    ) -> Result<()>
    where
        F: FnMut(Trial) -> Result<Vec<f64>>,
    {
        for _ in 0..n_trials {
            let trial = self.ask(sampler.clone())?;
            let trial_number = trial.number;

            // Call an objective function.
            let values = objective(trial);
            match values {
                Ok(values) => {
                    if self.directions.len() != values.len() {
                        return Err(Error::new(ErrorKind::InvalidObjectiveValues));
                    }
                    self.tell(trial_number, TrialStateValues::Complete(values))?;
                }
                Err(e) => {
                    self.tell(trial_number, TrialStateValues::Fail)?;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn get_trials(&self) -> Result<Vec<PersistedTrial>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let trials = guard.get_trials(self.id)?;
        Ok(trials.clone())
    }

    pub fn get_user_attr(&self, key: String) -> Result<Option<String>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let study = guard.get_study(self.id)?;

        let key = AttrKey::User(key.into());
        match study.attrs.get(&key) {
            Some(value) => Ok(Some(value.clone())),
            _ => Ok(None),
        }
    }

    pub fn set_user_attr(&self, attrs: HashMap<String, String>) -> Result<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let mut a = Attrs::new();
        for (key, value) in attrs {
            a.insert(AttrKey::User(key.into()), value);
        }
        guard.set_study_attrs(self.id, a, false)?;
        Ok(())
    }

    pub fn add_trial(&self, trial: PersistedTrial) -> Result<()> {
        trial.validate()?;
        if let TrialStateValues::Complete(ref values) = trial.state_values {
            if values.len() != self.directions.len() {
                return Err(Error::with_reason(
                    ErrorKind::InvalidObjectiveValues,
                    format!(
                        "The added trial has {} values, which is different from the number of objectives {} in the study.",
                        values.len(),
                        self.directions.len()
                    ),
                ));
            }
        }
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        guard.create_new_trial_from_template(self.id, &trial)?;
        Ok(())
    }

    pub fn enqueue_trial(
        &self,
        params: HashMap<String, CategoryLabel>,
        user_attrs: Option<Attrs>,
    ) -> Result<()> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let mut template = PersistedTrial::new(0, self.id, 0);
        template.state_values = TrialStateValues::Waiting;
        let fixed_attrs = fixed_params_to_attrs(&params);
        template.attrs.extend(fixed_attrs);

        if let Some(attrs) = user_attrs {
            template.attrs.extend(attrs);
        }

        let trial = guard.create_new_trial_from_template(self.id, &template)?;
        let trial_id = trial.id;
        drop(guard);

        let mut queue_guard = self.queue.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a queue guard: {e}"),
            )
        })?;
        queue_guard.push(trial_id)?;

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Clone)]
pub struct PersistedStudy {
    pub id: u32,
    pub name: String,
    pub directions: Vec<Direction>,
    pub attrs: Attrs,
}
impl PersistedStudy {
    pub fn new(id: u32, study_name: String, directions: Vec<Direction>) -> PersistedStudy {
        PersistedStudy {
            id,
            name: study_name,
            directions,
            attrs: Attrs::new(),
        }
    }
    // TODO(knshnb): Consider a builder pattern:
    // https://github.com/optuna/rustuna/pull/37#discussion_r1503510194
    pub fn new_with_attrs(
        id: u32,
        study_name: String,
        directions: Vec<Direction>,
        attrs: Attrs,
    ) -> PersistedStudy {
        PersistedStudy {
            id,
            name: study_name,
            directions,
            attrs,
        }
    }
}

pub fn get_best_trial(study: &Study) -> Result<u32> {
    let mut guard = study
        .storage
        .write()
        .map_err(|_| Error::new(ErrorKind::StorageError))?;
    let trials = guard.get_trials(study.id)?;

    let best_trial = trials
        .iter()
        .filter(|trial| matches!(trial.state_values, TrialStateValues::Complete(_)))
        .min_by(|a, b| {
            let a_value = match a.state_values {
                TrialStateValues::Complete(ref v) => {
                    assert!(v.len() == 1);
                    v[0]
                }
                _ => unreachable!("Unexpected state"),
            };
            let b_value = match b.state_values {
                TrialStateValues::Complete(ref v) => {
                    assert!(v.len() == 1);
                    v[0]
                }
                _ => unreachable!("Unexpected state"),
            };
            a_value
                .partial_cmp(&b_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or(Error::new(ErrorKind::NoCompletedTrial))?;
    Ok(best_trial.number)
}

// TODO(HideakiImamura): Support the faster algorithm for `len(directions) == 2`.
pub fn get_pareto_front(study: &Study) -> Result<Vec<u32>> {
    let mut guard = study
        .storage
        .write()
        .map_err(|_| Error::new(ErrorKind::StorageError))?;
    let trials = guard
        .get_trials(study.id)?
        .iter()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(ref _v)))
        .collect::<Vec<_>>();

    // TODO(HideakiImamura): Use Vec::with_capacity() to reduce the number of memory allocations.
    let mut pareto_front_numbers = vec![];
    trials.iter().for_each(|trial| {
        let mut dominated = false;
        let trial_values = match trial.state_values {
            TrialStateValues::Complete(ref v) => v,
            _ => panic!("Unexpected state"),
        };
        for other in trials.iter() {
            let other_values = match other.state_values {
                TrialStateValues::Complete(ref v) => v,
                _ => panic!("Unexpected state"),
            };
            if dominates(other_values, trial_values, &study.directions) {
                dominated = true;
                break;
            }
        }

        if !dominated {
            pareto_front_numbers.push(trial.number);
        }
    });

    Ok(pareto_front_numbers)
}

pub fn dominates(values0: &[f64], values1: &[f64], directions: &[Direction]) -> bool {
    assert_eq!(values0.len(), values1.len());
    assert_eq!(values0.len(), directions.len());

    let mut equal = true;
    for ((v0, v1), d) in values0.iter().zip(values1).zip(directions) {
        if *v0 != *v1 {
            equal = false;
        }
        let v1_dominate_v0 = match d {
            Direction::Minimize => *v0 > *v1,
            Direction::Maximize => *v0 < *v1,
        };
        if v1_dominate_v0 {
            return false; // Early return
        }
    }
    !equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::AttrKey;
    use crate::distribution::Distribution;
    use std::thread;

    use crate::sampler::RandomSampler;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study;
    use crate::study::get_best_trial;

    #[test]
    fn test_optimize() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                Ok(vec![value])
            },
            sampler,
            100,
        )?;
        assert!(get_best_trial(&study).is_ok());
        Ok(())
    }

    #[test]
    fn test_optimize_parallel() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy-study", storage, directions)?;

        thread::scope(|s| {
            for i in 0..4 {
                let study = study.clone();
                let sampler = Arc::new(Mutex::new(RandomSampler::seed_from_u64(i)));
                let choices = vec![String::from("foo"), String::from("bar")];
                s.spawn(move || {
                    study
                        .optimize(
                            |mut t| {
                                let x = t.suggest_float("x", 0.0, 10.0)?;
                                let y = t.suggest_float("y", 0.0, 10.0)?;
                                let z = t.suggest_int("z", 0, 10)?;
                                let _c = t.suggest_categorical("cat", &choices)?;
                                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                                Ok(vec![value])
                            },
                            sampler,
                            100,
                        )
                        .expect("Optimization failed");
                });
            }
        });
        assert!(get_best_trial(&study).is_ok());
        assert_eq!(study.get_trials()?.len(), 400);
        Ok(())
    }

    #[test]
    fn test_user_attr() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, directions)?;

        let mut trial = study.ask(sampler)?;
        trial.set_user_attr("key", String::from("bar"))?;
        let user_attr = trial
            .get_user_attr("key")
            .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
        assert_eq!(user_attr, "bar");
        Ok(())
    }

    #[test]
    fn test_get_best_trial() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                Ok(vec![value])
            },
            sampler,
            100,
        )?;

        let best_trial_number = get_best_trial(&study)?;
        assert!(best_trial_number < 100);
        Ok(())
    }

    #[test]
    fn test_get_pareto_front_trials() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize, Direction::Maximize];
        let study = create_study("dummy", storage, directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value0 = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                let value1 = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64) * 2.0;
                Ok(vec![value0, value1])
            },
            sampler,
            100,
        )?;

        let pareto_front_numbers = get_pareto_front(&study)?;
        assert!(!pareto_front_numbers.is_empty());
        assert!(pareto_front_numbers.len() <= 100);
        Ok(())
    }

    #[test]
    fn test_dominates() {
        let directions = vec![Direction::Minimize, Direction::Maximize];
        assert!(dominates(&[1.0, 2.0], &[2.0, 1.0], &directions));
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0], &directions));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0], &directions));
        assert!(!dominates(&[2.0, 1.0], &[2.0, 1.0], &directions));
    }

    #[test]
    fn test_dynamic_search_space() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, directions)?;

        study.optimize(
            |mut t| {
                t.suggest_float("x", 0.0, 10.0)?;
                Ok(vec![0.0])
            },
            sampler.clone(),
            5,
        )?;
        study.optimize(
            |mut t| {
                t.suggest_float("x", 1.0, 10.0)?;
                Ok(vec![0.0])
            },
            sampler.clone(),
            5,
        )?;
        Ok(())
    }

    #[test]
    fn test_invalid_dynamic_search_space() -> Result<()> {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, directions)?;

        study.optimize(
            |mut t| {
                t.suggest_float("x", 0.0, 10.0)?;
                Ok(vec![0.0])
            },
            sampler.clone(),
            5,
        )?;
        let error = study
            .optimize(
                |mut t| {
                    t.suggest_int("x", 0, 10)?;
                    Ok(vec![0.0])
                },
                sampler.clone(),
                5,
            )
            .unwrap_err();
        assert!(matches!(error.kind, ErrorKind::IncompatibleDistribution));
        Ok(())
    }

    #[test]
    fn test_add_trial_preserves_template_fields() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy-add-trial", storage, directions)?;

        let mut trial = PersistedTrial::new(999, 888, 777);
        trial.state_values = TrialStateValues::Complete(vec![1.23]);
        trial.datetime_start = Some("2026-04-02 03:04:05.678".to_string());
        trial.datetime_complete = Some("2026-04-02 03:14:15.678".to_string());
        trial.internal_params.insert("x".to_string(), 0.5);
        trial.distributions.insert(
            "x".to_string(),
            Distribution::Float {
                low: 0.0,
                high: 1.0,
                step: None,
                log: false,
            },
        );
        trial
            .attrs
            .insert(AttrKey::User("memo".into()), "\"imported\"".to_string());

        study.add_trial(trial)?;
        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].number, 0);
        assert_eq!(trials[0].study_id, study.id);
        assert_eq!(
            trials[0].datetime_start.as_deref(),
            Some("2026-04-02 03:04:05.678")
        );
        assert_eq!(
            trials[0].datetime_complete.as_deref(),
            Some("2026-04-02 03:14:15.678")
        );
        assert_eq!(trials[0].internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trials[0].attrs.get(&AttrKey::User("memo".into())),
            Some(&"\"imported\"".to_string())
        );
        Ok(())
    }
}
