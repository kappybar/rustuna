use std::sync::{Arc, Mutex};

use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, Direction};
use rustuna_core::Result;
use rustuna_importance::{self, PedAnovaImportanceEvaluator};

fn main() -> Result<()> {
    let storage = InMemoryStorage::new();
    let directions = vec![Direction::Minimize];
    let study = create_study("simple-quadratic", storage, directions)?;

    let sampler = Arc::new(Mutex::new(RandomSampler::new()));
    study.optimize(
        |mut t| {
            let x = t.suggest_float("x", 0.0, 10.0)?;
            let y = t.suggest_float("y", 0.0, 10.0)?;
            let value = x + y * 3.0;
            println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
            Ok(vec![value])
        },
        sampler,
        50,
    )?;
    let evaluator = PedAnovaImportanceEvaluator::new(0.1, 1.0, true);
    let importances = rustuna_importance::get_param_importances(&study, &evaluator)?;
    println!("Parameter importances: {importances:?}");

    Ok(())
}
