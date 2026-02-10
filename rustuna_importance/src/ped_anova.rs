use std::collections::HashMap;
use rustuna_core::trial::PersistedTrial;
use rustuna_core::distribution::Distribution;
use rustuna_core::study::{Study, Direction};
use crate::common::{self, ImportanceEvaluator};


pub struct PedAnovaImportanceEvaluator {
    target_quantile: f64,
    region_quantile: f64,
    evaluate_on_local: bool,
    n_steps: usize,
    prior_weight: f64,
    min_n_top_trials: usize,
}

impl PedAnovaImportanceEvaluator {
    pub fn new(target_quantile: f64, region_quantile: f64, evaluate_on_local: bool) -> Self {
        Self {
            target_quantile,
            region_quantile,
            evaluate_on_local,
            n_steps: 50,
            prior_weight: 1.0,
            min_n_top_trials: 2
        }
    }

    fn get_top_quantile_trials<'a>(
        &self,
        study: &Study,
        trials: &'a [PersistedTrial],
        quantile: f64,
        target: &dyn Fn(&PersistedTrial) -> f64,
    ) -> Vec<&'a PersistedTrial> {
        if quantile == 1.0 {
            return trials.iter().collect();
        }
        let is_lower_better = study.directions[0] == Direction::Minimize;
        let objective_values = trials.iter().map(|t| {
            let v = target(t);
            if is_lower_better { v } else { -v }
        }).collect::<Vec<_>>();
        let num_trials = trials.len();
        let num_top_trials = ((quantile * (num_trials as f64 - 1.0)).floor() as usize).min(num_trials - 1);

        let (_, &mut threshold, _) = objective_values.clone().select_nth_unstable_by(
            num_top_trials,
            |a, b| a.total_cmp(b),
        );
        let top_trials = trials.iter().zip(objective_values.iter())
            .filter(|(_, &v)| v <= threshold)
            .map(|(t, _)| t)
            .collect();
        top_trials
    }
}

impl ImportanceEvaluator for PedAnovaImportanceEvaluator {
    fn evaluate_with_target(&self, study: &Study, target: &dyn Fn(&PersistedTrial) -> f64) -> Result<HashMap<String, f64>> {
        let trials = common::get_filtered_trials(study, target)?;
        let dists = common::get_intersection_search_space(&trials);

        if trials.len() < self.min_n_top_trials {
            return Ok(dists.into_iter().map(|(name, _)| (name, 0.0)).collect());
        }

        let target_trials = self.get_top_quantile_trials(study, &trials, self.target_quantile, target);
        let region_trials = self.get_top_quantile_trials(study, &trials, self.region_quantile, target);

        let quantile = target_trials.len() as f64 / region_trials.len() as f64;

        let mut importances = dists.into_iter().map(|(name, dist)| {
            let importance = if dist.is_single() {
                0.0
            } else {
                quantile.powi(2) * self.compute_pearson_divergence(name, &dist, &target_trials, &region_trials)
            };
            (name, importance)
        }).collect::<HashMap<_, _>>();
        Ok(importances)
    }
}


fn count_numerical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[PersistedTrial],
    n_steps: usize,
) -> Vec<u32> {
    let (low, high, log, n_steps) = match dist {
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if *log {
                let log2_domain_size = ((high - low + 1) as f64).log2().ceil() as usize + 1;
                n_steps.min(log2_domain_size)
            } else {
                n_steps.min(((high - low) / step + 1) as usize)
            };
            (*low as f64, *high as f64, *log, n_steps)
        },
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if let Some(step) = step {
                n_steps.min(((high - low) / step).round() as usize + 1)
            } else {
                n_steps
            };
            (*low, *high, *log, n_steps)
        },
        _ => unreachable!("Invalid distribution type for numerical calculation"),
    };
    let (low, high) = if log {
        (low.ln(), high.ln())
    } else {
        (low, high)
    };
    let param_values = trials.iter().map(|t| {
        let v = t.internal_params[param_name];
        if log { v.ln() } else { v }
    });
    let step_size = (high - low) / (n_steps as f64);
    let mut counts = vec![0u32; n_steps];
    for v in param_values {
        let idx = ((v - low) / step_size).floor().max(0.0).min((n_steps - 1) as f64) as usize;
        counts[idx] += 1;
    }
    counts
}




fn count_categorical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[PersistedTrial],
) -> Vec<u32> {
    let cardinality = match dist {
        Distribution::Categorical { cardinality } => *cardinality,
        _ => unreachable!("Invalid distribution type for categorical calculation"),
    };
    let mut counts = vec![0u32; cardinality];
    for t in trials {
        let v = t.internal_params[param_name];
        counts[v as usize] += 1;
    }
    counts
}