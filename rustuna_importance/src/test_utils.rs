use std::sync::{Arc, Mutex};
use rustuna_core::sampler::RandomSampler;
use rustuna_core::trial::Trial;
use rustuna_core::Result;
use rustuna_core::distribution::Distribution;
use rustuna_core::study::{self, Direction, Study};
use rustuna_core::storage::InMemoryStorage;

fn single_objective(mut trial: Trial) -> Result<Vec<f64>> {
    let x1 = trial.suggest_float("x1", 0.1, 3.0)?;
    let x2 = trial.suggest("x2", &Distribution::Float {low: 0.1, high: 0.3, step: None, log: true})?;
    let x3 = trial.suggest("x3", &Distribution::Float {low: 2.0, high:4.0, step: None, log: true})?;
    Ok(vec![x1 + x2 * x3])
}

fn multi_objective(mut trial: Trial) -> Result<Vec<f64>> {
    let x1 = trial.suggest_float("x1", 0.1, 3.0)?;
    let x2 = trial.suggest("x2", &Distribution::Float {low: 0.1, high: 0.3, step: None, log: true})?;
    let x3 = trial.suggest("x3", &Distribution::Float {low: 2.0, high:4.0, step: None, log: true})?;
    Ok(vec![x1, x2 * x3])
}

pub(crate) fn get_study(seed: u64, n_trials: usize, is_multi_objective: bool, direction: Direction) -> Result<Study> {
    let storage = InMemoryStorage::new();
    let directions = if is_multi_objective {
        vec![direction.clone(), direction]
    } else {
        vec![direction]
    };
    let mut study = study::create_study("test-study", storage, directions)?;
    let sampler = Arc::new(Mutex::new(RandomSampler::seed_from_u64(seed)));
    study.optimize(
        if is_multi_objective {
            multi_objective
        } else {
            single_objective
        },
        sampler,
        n_trials,
    )?;
    Ok(study)
}