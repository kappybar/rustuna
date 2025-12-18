use core::panic;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustuna_core::distribution::Distribution;
use rustuna_core::parzen_estimator::ParzenEstimator;
use rustuna_core::sampler::{Context, RandomSampler, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

pub struct TpeConfig {
    pub multivariate: bool,
    pub n_startup_trials: usize,
    pub seed: Option<u64>,
}
impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            multivariate: true,
            n_startup_trials: 10,
            seed: None,
        }
    }
}

pub struct TpeSampler {
    rng: StdRng,
    multivariate: bool,
    n_startup_trials: usize,
    random_sampler: RandomSampler,
}
impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}
impl TpeSampler {
    pub fn from_config(cfg: TpeConfig) -> TpeSampler {
        let mut rng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(Default::default()),
        };
        let seed_for_random_sampler = rng.gen();
        Self {
            rng,
            multivariate: cfg.multivariate,
            n_startup_trials: cfg.n_startup_trials,
            random_sampler: RandomSampler::seed_from_u64(seed_for_random_sampler),
        }
    }

    pub fn new() -> TpeSampler {
        Self::from_config(TpeConfig::default())
    }

    pub fn seed_from_u64(seed: u64) -> TpeSampler {
        Self::from_config(TpeConfig {
            multivariate: false,
            seed: Some(seed),
            n_startup_trials: 10,
        })
    }

    fn sample(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let trials = guard.get_trials(ctx.study_id)?.clone();
        drop(guard);

        let complete_trials: Vec<_> = trials.into_iter().filter(|t| t.is_finished()).collect();
        let gamma = Self::gamma_for_single_objective(complete_trials.len());
        let direction = &ctx.directions[0];
        let (good_trials, poor_trials) =
            Self::split_trials_for_single_objective(&complete_trials, direction, gamma);
        let pe_good = Self::build_parzen_estimator(&good_trials, search_space);
        let pe_poor = Self::build_parzen_estimator(&poor_trials, search_space);

        let n_ei_candidates = 24;
        let samples_good = pe_good.sample(&mut self.rng, n_ei_candidates);
        let log_ei_values = samples_good
            .iter()
            .map(|s| pe_good.log_pdf(s) - pe_poor.log_pdf(s))
            .collect::<Vec<f64>>();
        let max_index = log_ei_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        Ok(samples_good[max_index].clone())
    }

    fn split_trials_for_single_objective(
        trials: &Vec<rustuna_core::trial::PersistedTrial>,
        direction: &Direction,
        gamma: usize,
    ) -> (
        Vec<rustuna_core::trial::PersistedTrial>,
        Vec<rustuna_core::trial::PersistedTrial>,
    ) {
        let mut sorted_trials = trials.clone();
        sorted_trials.sort_by(|a, b| {
            let a_value = match &a.state_values {
                TrialStateValues::Complete(v) => v[0],
                _ => panic!("Unexpected non-complete trial found during TPE sampling"),
            };
            let b_value = match &b.state_values {
                TrialStateValues::Complete(v) => v[0],
                _ => panic!("Unexpected non-complete trial found during TPE sampling"),
            };
            match direction {
                Direction::Minimize => a_value.partial_cmp(&b_value).unwrap(),
                Direction::Maximize => b_value.partial_cmp(&a_value).unwrap(),
            }
        });
        let good_trials = sorted_trials
            .iter()
            .take(gamma)
            .cloned()
            .collect::<Vec<_>>();
        let poor_trials = sorted_trials
            .iter()
            .skip(gamma)
            .cloned()
            .collect::<Vec<_>>();
        (good_trials, poor_trials)
    }

    fn build_parzen_estimator(
        trials: &Vec<rustuna_core::trial::PersistedTrial>,
        search_space: &HashMap<String, Distribution>,
    ) -> ParzenEstimator {
        let mut sorted_keys: Vec<_> = search_space.keys().collect();
        sorted_keys.sort();
        let mut observations: HashMap<String, Vec<f64>> = HashMap::new();
        for key in sorted_keys.iter() {
            let mut vals = Vec::new();
            for t in trials {
                if let Some(&v) = t.internal_params.get(*key) {
                    vals.push(v);
                }
            }
            observations.insert((*key).clone(), vals);
        }
        let n_weights = match observations.values().next() {
            None => 0,
            Some(first) => {
                let first_len = first.len();
                assert!(
                    observations.values().all(|v| v.len() == first_len),
                    "Observations have inconsistent lengths"
                );
                first_len
            }
        };
        let weights = Self::weights_for_single_objective(n_weights);
        let prior_weight = 1.0;
        ParzenEstimator::new(observations, search_space.clone(), weights, prior_weight)
    }

    fn gamma_for_single_objective(n: usize) -> usize {
        let threashold: usize = 25;
        let gamma = std::cmp::min(((0.1 * n as f64).ceil()) as usize, threashold);
        gamma
    }

    fn weights_for_single_objective(x: usize) -> Vec<f64> {
        let threashold = 25;
        if x == 0 {
            vec![]
        } else if x < threashold {
            vec![1.0; x]
        } else {
            let n = x - threashold;
            let start = 1.0 / (x as f64);
            if n == 0 {
                vec![1.0; threashold]
            } else if n == 1 {
                let mut v = Vec::with_capacity(threashold + 1);
                v.push(start);
                v.extend(std::iter::repeat_n(1.0, threashold));
                v
            } else {
                let step = (1.0 - start) / ((n - 1) as f64);
                let mut v = Vec::with_capacity(n + threashold);
                for i in 0..n {
                    v.push(start + (i as f64) * step);
                }
                v.extend(std::iter::repeat_n(1.0, threashold));
                v
            }
        }
    }
}

impl Sampler for TpeSampler {
    fn sample_independent(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if ctx.directions.len() != 1 {
            return Err(Error::new(ErrorKind::UnsupportedMultiObjective));
        }

        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let trials = guard.get_trials(ctx.study_id)?.clone();
        drop(guard);

        let complete_trials: Vec<_> = trials.into_iter().filter(|t| t.is_finished()).collect();
        if complete_trials.len() < self.n_startup_trials {
            let params =
                self.random_sampler
                    .sample_independent(ctx, storage, name, distribution)?;
            return Ok(params);
        }

        let search_space = HashMap::from([(name.to_string(), distribution.clone())]);
        let params = self.sample(ctx, storage, &search_space)?;
        Ok(params[name])
    }

    fn support_joint_sampling(&self) -> bool {
        self.multivariate
    }

    fn sample_joint(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if ctx.directions.len() != 1 {
            return Err(Error::new(ErrorKind::UnsupportedMultiObjective));
        }

        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let trials = guard.get_trials(ctx.study_id)?.clone();
        drop(guard);

        let complete_trials: Vec<_> = trials.into_iter().filter(|t| t.is_finished()).collect();
        if complete_trials.len() < self.n_startup_trials {
            let params = HashMap::new();
            return Ok(params);
        }
        let params = self.sample(ctx, storage, search_space)?;
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::get_best_trial;
    use rustuna_core::study::{create_study, Direction};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_optimize() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let mut study = create_study("simple-quadratic", storage, directions).unwrap();

        let sampler = Arc::new(Mutex::new(TpeSampler::new()));
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_int("y", 0, 10)?;
                    let z = *t.suggest_categorical("z", &[1, 2, 3, 4, 5])? as i64;
                    let value = (x - 3.0).powi(2) + (y - 5).pow(2) as f64 + (z - 2).pow(2) as f64;
                    println!(
                        "{:2} x: {}, y: {}, z: {}, value: {}",
                        t.number, x, y, z, value
                    );
                    Ok(vec![value])
                },
                sampler,
                20,
            )
            .unwrap();
        let best_trial_number = get_best_trial(&study);
        assert!(best_trial_number.is_ok());
    }
}
