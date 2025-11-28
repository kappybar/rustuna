use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::prelude::*;
use rand::rngs::StdRng;
use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{Context, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

pub struct TpeSampler {
    rng: StdRng,
}
impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl TpeSampler {
    pub fn new() -> TpeSampler {
        TpeSampler {
            rng: StdRng::from_seed(Default::default()),
        }
    }
    pub fn seed_from_u64(seed: u64) -> TpeSampler {
        TpeSampler {
            rng: StdRng::seed_from_u64(seed),
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

        // TODO(c-bata): Support step and logs.
        let tpe_param = match distribution {
            Distribution::Float {
                low,
                high,
                step: _,
                log: _,
            } => tpe::range(*low, *high),
            Distribution::Int {
                low,
                high,
                step: _,
                log: _,
            } => tpe::range(*low as f64, *high as f64),
            Distribution::Categorical { cardinality } => tpe::categorical_range(*cardinality),
        }
        .map_err(|_e| Error::new(ErrorKind::SamplerError))?;
        let mut optimizer = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe_param);

        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let trials = guard.get_trials(ctx.study_id)?;
        let direction = &ctx.directions[0];

        for t in trials {
            let mut value = match t.state_values {
                TrialStateValues::Complete(ref values) => {
                    if let Some(value) = values.first() {
                        *value
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            if *direction == Direction::Maximize {
                value = -value;
            }
            let param = if let Some(param) = t.internal_params.get(name) {
                *param
            } else {
                continue;
            };
            // TODO(c-bata): Propagate the original error context.
            optimizer
                .tell(param, value)
                .map_err(|_| Error::new(ErrorKind::SamplerError))?;
        }

        let param = optimizer
            .ask(&mut self.rng)
            .map_err(|_e| Error::new(ErrorKind::SamplerError))?;
        Ok(param)
    }

    fn support_joint_sampling(&self) -> bool {
        false
    }

    fn sample_joint(
        &mut self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        unreachable!()
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
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
                    println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
                    Ok(vec![value])
                },
                sampler,
                50,
            )
            .unwrap();
        let best_trial_number = get_best_trial(&study);
        assert!(best_trial_number.is_ok());
    }
}
