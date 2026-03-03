use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::distribution::Distribution;
use crate::storage::Storage;
use crate::study::Direction;
use crate::Result;

#[derive(Debug, Clone)]
pub struct Context {
    pub study_id: u32,
    pub directions: Vec<Direction>,
    pub trial_number: u32,
    pub trial_id: u32,
}

pub trait Sampler: Send {
    fn sample_independent(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64>;
    fn support_joint_sampling(&self) -> bool;
    fn sample_joint(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>>;
}

pub struct RandomSampler {
    rng: StdRng,
}
impl Default for RandomSampler {
    fn default() -> Self {
        RandomSampler::new()
    }
}
impl RandomSampler {
    pub fn new() -> RandomSampler {
        RandomSampler {
            rng: StdRng::from_seed(Default::default()),
        }
    }

    pub fn seed_from_u64(seed: u64) -> RandomSampler {
        RandomSampler {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

fn round_to_step(value: f64, low: f64, high: f64, step: f64) -> f64 {
    let mut stepped = low + ((value - low) / step).round() * step;
    if stepped < low {
        stepped = low;
    }
    if stepped > high {
        stepped = high;
    }
    stepped
}

fn sample_float_with_step(
    rng: &mut StdRng,
    low: f64,
    high: f64,
    step: Option<f64>,
    log: bool,
) -> f64 {
    match (step, log) {
        (None, false) => rng.gen_range(low..high),
        (None, true) => rng.gen_range(low.ln()..high.ln()).exp(),
        (Some(step), false) => {
            let max_index = ((high - low) / step).floor().max(0.0) as i64;
            let index = rng.gen_range(0..=max_index);
            low + (index as f64) * step
        }
        (Some(step), true) => {
            let value = rng.gen_range(low.ln()..high.ln()).exp();
            round_to_step(value, low, high, step)
        }
    }
}

fn sample_int_with_step(rng: &mut StdRng, low: i64, high: i64, step: i64, log: bool) -> f64 {
    let low_f = low as f64;
    let high_f = high as f64;
    let step_f = step as f64;
    if log {
        let value = rng.gen_range(low_f.ln()..high_f.ln()).exp();
        let max_index = ((high_f - low_f) / step_f).floor().max(0.0) as i64;
        let mut index = ((value - low_f) / step_f).round() as i64;
        if index < 0 {
            index = 0;
        }
        if index > max_index {
            index = max_index;
        }
        low_f + (index as f64) * step_f
    } else {
        let max_index = ((high - low) / step).max(0) as i64;
        let index = rng.gen_range(0..=max_index);
        (low + index * step) as f64
    }
}
impl Sampler for RandomSampler {
    fn sample_independent(
        &mut self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        match distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let value = sample_float_with_step(&mut self.rng, *low, *high, *step, *log);
                Ok(value)
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                let value = sample_int_with_step(&mut self.rng, *low, *high, *step, *log);
                Ok(value)
            }
            Distribution::Categorical { cardinality } => {
                let param_value = self.rng.gen_range(0..*cardinality);
                Ok(param_value as f64)
            }
        }
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

    use std::sync::Mutex;

    use crate::storage::InMemoryStorage;
    use crate::study::create_study;
    use crate::trial::Trial;

    pub struct DummyJointSampler {
        joint_params: HashMap<String, f64>,
    }
    impl Sampler for DummyJointSampler {
        fn sample_independent(
            &mut self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn Storage>>,
            _name: &str,
            _distribution: &Distribution,
        ) -> Result<f64> {
            Ok(0.0)
        }

        fn support_joint_sampling(&self) -> bool {
            true
        }

        fn sample_joint(
            &mut self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn Storage>>,
            _search_space: &HashMap<String, Distribution>,
        ) -> Result<HashMap<String, f64>> {
            Ok(self.joint_params.clone())
        }
    }

    fn objective(mut t: Trial) -> Result<Vec<f64>> {
        let x = t.suggest_float("x", -10.0, 10.0)?;
        let y = t.suggest_float("y", -10.0, 10.0)?;
        Ok(vec![x * x + y * y])
    }

    #[test]
    fn test_joint_sampling_empty() -> Result<()> {
        let joint_params = HashMap::new();
        let sampler = Arc::new(Mutex::new(DummyJointSampler { joint_params }));
        let mut study = create_study("dummy", InMemoryStorage::new(), vec![Direction::Minimize])?;
        study.optimize(objective, sampler, 2)?;
        Ok(())
    }

    #[test]
    fn test_joint_sampling_partially() -> Result<()> {
        let mut joint_params = HashMap::new();
        joint_params.insert(String::from("x"), 0.5);

        let sampler = Arc::new(Mutex::new(DummyJointSampler { joint_params }));
        let mut study = create_study("dummy", InMemoryStorage::new(), vec![Direction::Minimize])?;
        study.optimize(objective, sampler, 2)?;

        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[1].internal_params["x"], 0.5);
        assert!(trials[1].internal_params.contains_key("y"));
        Ok(())
    }

    #[test]
    fn test_joint_sampling_all() -> Result<()> {
        let mut joint_params = HashMap::new();
        joint_params.insert(String::from("x"), 1.0);
        joint_params.insert(String::from("y"), 1.0);

        let sampler = Arc::new(Mutex::new(DummyJointSampler { joint_params }));
        let mut study = create_study("dummy", InMemoryStorage::new(), vec![Direction::Minimize])?;
        study.optimize(objective, sampler, 2)?;

        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[1].internal_params["x"], 1.0);
        assert_eq!(trials[1].internal_params["y"], 1.0);
        Ok(())
    }
}
