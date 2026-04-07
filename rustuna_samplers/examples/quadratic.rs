// How to run this example:
// $ cargo run --example quadratic

use std::sync::{Arc, Mutex};

use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::get_best_trial;
use rustuna_core::study::{create_study, Direction};
use rustuna_core::Result;
use rustuna_samplers::tpe::TpeSampler;

fn main() -> Result<()> {
    let storage = InMemoryStorage::new();
    let directions = vec![Direction::Minimize];
    let study = create_study("simple-quadratic", storage, directions)?;

    let sampler = Arc::new(Mutex::new(TpeSampler::new()));
    study.optimize(
        |mut t| {
            let x = t.suggest_float("x", 0.0, 10.0)?;
            let y = t.suggest_int("y", 0, 10)?;
            let value = (x - 3.0).powi(2) + (y - 5).pow(2) as f64;
            println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
            Ok(vec![value])
        },
        sampler,
        50,
    )?;

    let best_trial_number = get_best_trial(&study)?;
    let trial = study.get_trials()?[best_trial_number as usize].clone();
    println!("Best trial: {trial:?}");
    Ok(())
}
