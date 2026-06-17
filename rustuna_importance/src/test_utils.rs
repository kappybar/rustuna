use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{self, Direction, Study};
use rustuna_core::trial::Trial;
use rustuna_core::Result;

pub(crate) enum ObjectiveType {
    Single,
    Multi,
    Conditional,
}

fn single_objective(mut trial: Trial) -> Result<Vec<f64>> {
    let x1 = trial.suggest_float("x1", 0.1, 3.0)?;
    let x2 = trial.suggest(
        "x2",
        &Distribution::Float {
            low: 0.1,
            high: 0.3,
            step: None,
            log: true,
        },
    )?;
    let x3 = trial.suggest(
        "x3",
        &Distribution::Float {
            low: 1.0,
            high: 3.0,
            step: Some(1.0),
            log: true,
        },
    )?;
    let x4 = trial.suggest_int("x4", -3, 3)?;
    let x5 = trial.suggest(
        "x5",
        &Distribution::Int {
            low: 1,
            high: 3,
            step: 1,
            log: true,
        },
    )?;
    let x6 = trial.suggest_categorical("x6", &[1.0, 1.1, 1.2])?;
    Ok(vec![x1.powi(4) + x2 + x3 - x4.pow(2) as f64 - x5 + x6])
}

fn multi_objective(mut trial: Trial) -> Result<Vec<f64>> {
    let x1 = trial.suggest_float("x1", 0.1, 3.0)?;
    let x2 = trial.suggest(
        "x2",
        &Distribution::Float {
            low: 0.1,
            high: 0.3,
            step: None,
            log: true,
        },
    )?;
    let x3 = trial.suggest(
        "x3",
        &Distribution::Float {
            low: 0.1,
            high: 3.0,
            step: Some(1.0),
            log: true,
        },
    )?;
    let x4 = trial.suggest_int("x4", -3, 3)?;
    let x5 = trial.suggest(
        "x5",
        &Distribution::Int {
            low: 1,
            high: 3,
            step: 1,
            log: true,
        },
    )?;
    let x6 = trial.suggest_categorical("x6", &[1.0, 1.1, 1.2])?;
    Ok(vec![x1.powi(4) + x2 + x3, x4.pow(2) as f64 - x5 + x6])
}

fn conditional_objective(mut trial: Trial) -> Result<Vec<f64>> {
    let c = trial.suggest_float("c", 0.0, 1.0)?;
    if c < 0.5 {
        let x = trial.suggest_float("x", 0.0, 10.0)?;
        Ok(vec![x])
    } else {
        let y = trial.suggest_float("y", -10.0, 0.0)?;
        Ok(vec![y])
    }
}

pub(crate) fn get_study(
    seed: u64,
    n_trials: usize,
    objective_type: ObjectiveType,
    direction: Direction,
) -> Result<Study> {
    let storage = InMemoryStorage::new();
    let directions = match objective_type {
        ObjectiveType::Multi => vec![direction.clone(), direction],
        _ => vec![direction],
    };
    let study = study::create_study(
        "test-study",
        storage,
        RandomSampler::seed_from_u64(seed),
        directions,
    )?;
    study.optimize(
        match objective_type {
            ObjectiveType::Single => single_objective,
            ObjectiveType::Multi => multi_objective,
            ObjectiveType::Conditional => conditional_objective,
        },
        n_trials,
    )?;
    Ok(study)
}
