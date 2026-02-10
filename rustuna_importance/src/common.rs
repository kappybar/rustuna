use rustuna_core::study::Study;
use std::collections::HashMap;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use rustuna_core::distribution::Distribution;

pub struct ImportanceOptions<'a> {
    // NOTE(kAIto47802): Currently, the `param` argument is not implemented.
    // We plan to implement it when we support condPED-ANOVA:
    // - https://arxiv.org/abs/2601.20800
    pub target: Option<&'a dyn Fn(&PersistedTrial) -> f64>,
    pub normalize: bool,
}

impl<'a> Default for ImportanceOptions<'a> {
    fn default() -> Self {
        Self { target: None, normalize: true }
    }
}


impl<'a> ImportanceOptions<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_target(mut self, target: &'a dyn Fn(&PersistedTrial) -> f64) -> Self {
        self.target = Some(target);
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

    fn evaluate(&self, study: &Study) -> Result<HashMap<String, f64>> {
        self.evaluate_with_target(study, &|t| {
            match &t.state_values {
                TrialStateValues::Complete(values) => {
                    assert_eq!(values.len(), 1, "Specify the `target` function for multi-objective studies.");
                    values[0]
                },
                _ => unreachable!("Only completed trials should be evaluated."),
            }
        })
    }
    fn evaluate_with_target(&self, study: &Study, target: &dyn Fn(&PersistedTrial) -> f64) -> Result<HashMap<String, f64>>;
}

pub(crate) fn get_filtered_trials(study: &Study, target: &dyn Fn(&PersistedTrial) -> f64) -> Result<Vec<PersistedTrial>> {
    let mut guard = study
        .storage
        .write()
        .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
    // TODO(c-bata): Avoid to clone trials.
    let completed_trials = guard
        .get_trials(study.id)?
        .iter()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(_)))
        .filter(|t| target(t).is_finite())
        .cloned()
        .collect::<Vec<_>>();
    if completed_trials.is_empty() {
        Err(Error::new(ErrorKind::NoCompletedTrial))
    } else {
        Ok(completed_trials)
    }
}

pub(crate) fn get_intersection_search_space(trials: &[PersistedTrial]) -> HashMap<String, Distribution> {
    let mut intersection_search_space = trials[0].distributions.clone();
    for trial in &trials[1..] {
        intersection_search_space.retain(|k, v| {
            trial.distributions.get(k).is_some_and(|v2| v2 == v)
        });
    }
    intersection_search_space
}