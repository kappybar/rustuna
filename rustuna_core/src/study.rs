use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::attr::{AttrKey, Attrs};
use crate::distribution::Distribution;
use crate::sampler::{Context as SamplerContext, Sampler};
use crate::storage::Storage;
use crate::trial::{PersistedTrial, Trial, TrialStateValues};
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
}
impl Study {
    pub fn new(
        id: u32,
        name: String,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
    ) -> Self {
        Study {
            id,
            name,
            directions,
            storage,
        }
    }

    pub fn from_id(id: u32, storage: Arc<RwLock<dyn Storage>>) -> Result<Self> {
        let mut guard = storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let study = guard.get_study(id)?;
        let name = study.name.clone();
        let directions = study.directions.clone();
        drop(guard);
        Ok(Study::new(id, name, directions, storage))
    }

    pub fn from_name(name: String, storage: Arc<RwLock<dyn Storage>>) -> Result<Self> {
        let mut guard = storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
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

    pub fn ask(&mut self, sampler: Arc<Mutex<dyn Sampler>>) -> Result<Trial> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let trial_number = guard.create_new_trial(self.id)?.number;
        drop(guard);

        // Joint sampling
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
                directions: self.directions.clone(),
            };
            let params = guard.sample_joint(&ctx, self.storage.clone(), &joint_search_space)?;
            let mut joint_params = HashMap::new();
            for (name, param_value) in params {
                if !joint_search_space.contains_key(&name) {
                    continue; // ignore parameters not in search space
                }
                let distribution = joint_search_space[&name].clone();
                joint_params.insert(name, (distribution, param_value));
            }
            joint_params
        } else {
            HashMap::new()
        };

        let trial = Trial::new(
            self.id,
            trial_number,
            self.directions.clone(),
            Arc::clone(&self.storage),
            sampler.clone(),
            joint_params,
        );
        Ok(trial)
    }

    pub fn tell(&mut self, trial_number: u32, state_values: TrialStateValues) -> Result<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        guard.set_trial_state_values(self.id, trial_number, state_values)?;
        Ok(())
    }

    pub fn optimize<F>(
        self: &mut Study,
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
        let mut guard = self.storage.write().unwrap();
        let trials = guard.get_trials(self.id)?;
        Ok(trials.clone())
    }

    pub fn get_user_attr(&self, key: String) -> Result<Option<String>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let study = guard.get_study(self.id)?;

        let key = AttrKey::User(key);
        match study.attrs.get(&key) {
            Some(value) => Ok(Some(value.clone())),
            _ => Ok(None),
        }
    }

    pub fn set_user_attr(&mut self, attrs: HashMap<String, String>) -> Result<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let mut a = Attrs::new();
        for (key, value) in attrs {
            a.insert(AttrKey::User(key), value);
        }
        guard.set_study_attrs(self.id, a, false)?;
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
    let mut guard = study.storage.write().unwrap();
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
            a_value.partial_cmp(&b_value).unwrap()
        })
        .ok_or(Error::new(ErrorKind::NoCompletedTrial))?;
    Ok(best_trial.number)
}

// TODO(HideakiImamura): Support the faster algorithm for `len(directions) == 2`.
pub fn get_pareto_front(study: &Study) -> Result<Vec<u32>> {
    let mut guard = study.storage.write().unwrap();
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
    use std::thread;

    use crate::sampler::RandomSampler;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study;
    use crate::study::get_best_trial;

    #[test]
    fn test_optimize() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let z = t.suggest_int("z", 0, 10)?;

                    let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                    Ok(vec![value])
                },
                sampler,
                100,
            )
            .unwrap();
        assert!(get_best_trial(&study).is_ok());
    }

    #[test]
    fn test_optimize_parallel() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy-study", storage, directions).unwrap();

        thread::scope(|s| {
            for i in 0..4 {
                let mut study = study.clone();
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
                        .unwrap();
                });
            }
        });
        assert!(get_best_trial(&study).is_ok());
        assert!(study.get_trials().unwrap().len() == 400);
    }

    #[test]
    fn test_user_attr() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        let mut trial = study.ask(sampler).unwrap();
        trial.set_user_attr("key", String::from("bar")).unwrap();
        let user_attr = trial.get_user_attr("key").unwrap();
        assert_eq!(user_attr, "bar");
    }

    #[test]
    fn test_get_best_trial() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let z = t.suggest_int("z", 0, 10)?;

                    let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                    Ok(vec![value])
                },
                sampler,
                100,
            )
            .unwrap();

        let best_trial_number = get_best_trial(&study);
        assert!(best_trial_number.is_ok());
        let best_trial_number = best_trial_number.unwrap();
        assert!(best_trial_number < 100);
    }

    #[test]
    fn test_get_pareto_front_trials() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize, Direction::Maximize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
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
            )
            .unwrap();

        let pareto_front_numbers = get_pareto_front(&study).unwrap();
        assert!(!pareto_front_numbers.is_empty());
        assert!(pareto_front_numbers.len() <= 100);
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
    fn test_dynamic_search_space() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    t.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![0.0])
                },
                sampler.clone(),
                5,
            )
            .unwrap();
        study
            .optimize(
                |mut t| {
                    t.suggest_float("x", 1.0, 10.0)?;
                    Ok(vec![0.0])
                },
                sampler.clone(),
                5,
            )
            .unwrap();
    }

    #[test]
    fn test_invalid_dynamic_search_space() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    t.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![0.0])
                },
                sampler.clone(),
                5,
            )
            .unwrap();
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
    }
}
