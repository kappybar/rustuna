use rustuna_core::study::Study;
use std::collections::HashMap;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use rustuna_core::distribution::Distribution;

pub trait ImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> HashMap<String, f64>;
    fn evaluate_with_target(&self, study: &Study, target: &dyn Fn(&PersistedTrial) -> f64) -> HashMap<String, f64>;
}

fn get_filtered_trials(study: &Study, target: &dyn Fn(&PersistedTrial) -> f64) -> Result<Vec<PersistedTrial>> {
    let mut guard = study
        .storage
        .write()
        .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
    // TODO(c-bata): Avoid to clone trials.
    let completed_trials = guard
        .get_trials(study.id)?
        .clone()
        .into_iter()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(_)))
        .filter(|t| target(t).is_finite())
        .collect::<Vec<_>>();
    if completed_trials.is_empty() {
        Err(Error::new(ErrorKind::NoCompletedTrial))
    } else {
        Ok(completed_trials)
    }
}

fn get_intersection_search_space(trials: &[PersistedTrial]) -> HashMap<String, Distribution> {
    let mut intersection_search_space = trials[0].distributions.clone();
    for trial in &trials[1..] {
        intersection_search_space.retain(|k, v| {
            trial.distributions.get(k).is_some_and(|v2| v2 == v)
        });
    }
    intersection_search_space
}