use core::panic;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::multi_objective;
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

const EPS: f64 = 1e-12;

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
    // TODO(y0z): Change to LruCache<(Vec<&PersistedTrial>, usize), (Vec<&PersistedTrial>, Vec<&PersistedTrial>)>
    split_cache: HashMap<(Vec<u32>, usize), (Vec<u32>, Vec<u32>)>,
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
            split_cache: HashMap::new(),
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
        let is_multi_objective = ctx.directions.len() > 1;

        let (pe_good, pe_poor) = if !is_multi_objective {
            let gamma = Self::gamma_for_single_objective(complete_trials.len());
            let direction: &Direction = &ctx.directions[0];
            let (good_trials, poor_trials) =
                Self::split_trials_for_single_objective(&complete_trials, direction, gamma);
            (
                Self::build_parzen_estimator(&good_trials, search_space, is_multi_objective),
                Self::build_parzen_estimator(&poor_trials, search_space, is_multi_objective),
            )
        } else {
            let directions: &[Direction] = &ctx.directions;
            let gamma = Self::gamma_for_multi_objective(complete_trials.len());
            let complete_trial_numbers = complete_trials
                .iter()
                .map(|t| t.number)
                .collect::<Vec<u32>>();
            let split_cache_key = (complete_trial_numbers.clone(), gamma);
            let (good_trials, poor_trials) = if self.split_cache.contains_key(&split_cache_key)
            {
                let (good_nums, poor_nums) =
                    self.split_cache.get(&split_cache_key).unwrap();
                let good_trials = complete_trials
                    .iter()
                    .filter(|t| good_nums.contains(&t.number))
                    .cloned()
                    .collect::<Vec<_>>();
                let poor_trials = complete_trials
                    .iter()
                    .filter(|t| poor_nums.contains(&t.number))
                    .cloned()
                    .collect::<Vec<_>>();
                (good_trials, poor_trials)
            } else {
                let (good_trials, poor_trials) = Self::split_trials_for_multi_objective(&complete_trials, directions, gamma);
                let good_nums = good_trials.iter().map(|t| t.number).collect();
                let poor_nums = poor_trials.iter().map(|t| t.number).collect();
                // We only cache the most recent split
                self.split_cache.clear();
                self.split_cache.insert(split_cache_key, (good_nums, poor_nums));
                (good_trials, poor_trials)
            };
            (
                Self::build_parzen_estimator(&good_trials, search_space, is_multi_objective),
                Self::build_parzen_estimator(&poor_trials, search_space, is_multi_objective),
            )
        };

        let n_ei_candidates = 24;
        let samples_good = pe_good.sample(&mut self.rng, n_ei_candidates);
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, s) in samples_good.iter().enumerate() {
            let acquisition = pe_good.log_pdf(s) - pe_poor.log_pdf(s);
            if acquisition > best_val {
                best_val = acquisition;
                best_idx = i;
            }
        }
        Ok(samples_good[best_idx].clone())
    }

    fn split_trials_for_single_objective(
        trials: &[rustuna_core::trial::PersistedTrial],
        direction: &Direction,
        gamma: usize,
    ) -> (
        Vec<rustuna_core::trial::PersistedTrial>,
        Vec<rustuna_core::trial::PersistedTrial>,
    ) {
        let n = trials.len();
        assert!(
            gamma <= n,
            "gamma must be less than or equal to the number of trials"
        );

        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        if gamma == n {
            return (trials.to_vec(), Vec::new());
        }

        fn value_for(t: &rustuna_core::trial::PersistedTrial) -> f64 {
            match &t.state_values {
                TrialStateValues::Complete(v) => v[0],
                _ => panic!("Unexpected non-complete trial found during TPE sampling"),
            }
        }

        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(gamma, |&i, &j| {
            let vi = value_for(&trials[i]);
            let vj = value_for(&trials[j]);
            let ord = vi.partial_cmp(&vj).unwrap_or(Ordering::Equal);
            match direction {
                Direction::Minimize => ord,
                Direction::Maximize => ord.reverse(),
            }
        });

        let mut good_trials = Vec::with_capacity(gamma);
        let mut poor_trials = Vec::with_capacity(n - gamma);
        for &i in idx.iter().take(gamma) {
            good_trials.push(trials[i].clone());
        }
        for &i in idx.iter().skip(gamma) {
            poor_trials.push(trials[i].clone());
        }
        (good_trials, poor_trials)
    }

    fn split_trials_for_multi_objective(
        trials: &[rustuna_core::trial::PersistedTrial],
        directions: &[Direction],
        gamma: usize,
    ) -> (
        Vec<rustuna_core::trial::PersistedTrial>,
        Vec<rustuna_core::trial::PersistedTrial>,
    ) {
        let n = trials.len();
        assert!(
            gamma <= n,
            "gamma must be less than or equal to the number of trials"
        );

        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        if gamma == n {
            return (trials.to_vec(), Vec::new());
        }

        // Assume minimization (negate value if maximization)
        let loss_values = trials
            .iter()
            .map(|t| {
                let vals = match &t.state_values {
                    TrialStateValues::Complete(v) => v.clone(),
                    _ => panic!("Unexpected non-complete trial found during TPE sampling"),
                };
                vals.iter()
                    .zip(directions.iter())
                    .map(|(&val, dir)| match dir {
                        Direction::Minimize => val,
                        Direction::Maximize => -val,
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        let nondomination_ranks = multi_objective::fast_non_dominated_sort(&loss_values);
        let mut rank_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &rank) in nondomination_ranks.iter().enumerate() {
            rank_to_indices.entry(rank).or_default().push(i);
        }

        let mut good_trials = Vec::with_capacity(gamma);
        let mut poor_trials = Vec::with_capacity(n - gamma);

        let mut current_rank = 0usize;
        while good_trials.len() + rank_to_indices.get(&current_rank).map_or(0, |v| v.len()) <= gamma
        {
            if let Some(indices) = rank_to_indices.get(&current_rank) {
                for &i in indices.iter() {
                    good_trials.push(trials[i].clone());
                }
            }
            current_rank += 1;
        }
        let hss_subset_size = gamma - good_trials.len();
        if hss_subset_size > 0 {
            let rank_i_loss_vals = rank_to_indices
                .get(&current_rank)
                .unwrap()
                .iter()
                .map(|&i| loss_values[i].as_slice())
                .collect::<Vec<&[f64]>>();
            let rank_i_indices = rank_to_indices.get(&current_rank).unwrap();

            let n_dims = directions.len();
            let mut worst_point = vec![f64::NEG_INFINITY; n_dims];
            for loss_val in rank_i_loss_vals.iter() {
                for d in 0..n_dims {
                    if loss_val[d] > worst_point[d] {
                        worst_point[d] = loss_val[d];
                    }
                }
            }
            let mut reference_point = Vec::with_capacity(n_dims);
            for &w in worst_point.iter() {
                let r = (1.1 * w).max(0.9 * w);
                reference_point.push(if r == 0.0 { EPS } else { r });
            }
            let selected_indices = multi_objective::hypervolume_subset_selection(
                &rank_i_loss_vals,
                rank_i_indices,
                &reference_point,
                hss_subset_size,
            );
            for &i in selected_indices.iter() {
                good_trials.push(trials[i].clone());
            }
        }
        for trial in trials.iter() {
            if !good_trials.iter().any(|t| t.number == trial.number) {
                poor_trials.push(trial.clone());
            }
        }

        (good_trials, poor_trials)
    }

    fn gamma_for_single_objective(n: usize) -> usize {
        let threashold: usize = 25;

        std::cmp::min(((0.1 * n as f64).ceil()) as usize, threashold)
    }

    fn gamma_for_multi_objective(n: usize) -> usize {
        (0.1 * n as f64).ceil() as usize
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

    fn weights_for_multi_objective(x: usize) -> Vec<f64> {
        // TODO(y0z): Implement more sophisticated weight calculation for multi-objective
        vec![1.0; x]
    }

    fn build_parzen_estimator(
        trials: &[rustuna_core::trial::PersistedTrial],
        search_space: &HashMap<String, Distribution>,
        is_multi_objective: bool,
    ) -> ParzenEstimator {
        let mut sorted_keys: Vec<&String> = search_space.keys().collect();
        sorted_keys.sort();
        let mut observations: HashMap<String, Vec<f64>> = HashMap::with_capacity(sorted_keys.len());

        // Handle conditional parameters
        let mut active_indices = Vec::with_capacity(trials.len());
        for (idx, key) in sorted_keys.iter().enumerate() {
            let mut vals = Vec::with_capacity(trials.len());
            for t in trials.iter() {
                if let Some(&v) = t.internal_params.get(*key) {
                    vals.push(v);
                    active_indices.push(idx);
                }
            }
            observations.insert((*key).clone(), vals);
        }
        let weights = if !is_multi_objective {
            Self::weights_for_single_objective(trials.len())
        } else {
            Self::weights_for_multi_objective(trials.len())
        };
        let active_weights: Vec<f64> = active_indices.iter().map(|&i| weights[i]).collect();
        let prior_weight = 1.0;
        ParzenEstimator::new(&observations, search_space, &active_weights, prior_weight)
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
    use rustuna_core::study::{create_study, Direction};
    use rustuna_core::study::{get_best_trial, get_pareto_front};
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

    #[test]
    fn test_multi_objective() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let mut study = create_study("simple-bi-objective", storage, directions).unwrap();

        let sampler = Arc::new(Mutex::new(TpeSampler::new()));
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let values = vec![
                        (x - 3.0).powi(2) + (y - 5.0).powi(2),
                        (x - 7.0).powi(2) + (y - 2.0).powi(2),
                    ];
                    println!("{:2} x: {}, y: {}, values: {:?}", t.number, x, y, values);
                    Ok(values)
                },
                sampler,
                20,
            )
            .unwrap();
        let best_trial_numbers = get_pareto_front(&study);
        assert!(best_trial_numbers.is_ok());
    }
}
