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

pub fn get_param_importances(study: &Study, evaluator: &impl ImportanceEvaluator) -> Result<HashMap<String, f64>> {
    get_param_importances_with(study, evaluator, ImportanceOptions::default())
}

pub fn get_param_importances_with(study: &Study, evaluator: &impl ImportanceEvaluator, opts: ImportanceOptions<'_>) -> Result<HashMap<String, f64>> {
    let normalize = opts.normalize;
    let importances = evaluator.evaluate_with(study, opts)?;
    if normalize { Ok(normalize_importances(importances)) } else { Ok(importances) }
}

pub trait ImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> Result<HashMap<String, f64>> {
        self.evaluate_with(study, ImportanceOptions::default())
    }
    fn evaluate_with(&self, study: &Study, opts: ImportanceOptions<'_>) -> Result<HashMap<String, f64>>;
}

fn normalize_importances(importances: HashMap<String, f64>) -> HashMap<String, f64> {
    let total = importances.values().sum::<f64>();
    if total == 0.0 {
        let n_params = importances.len() as f64;
        importances.into_keys().map(|k| (k, 1.0 / n_params)).collect()
    } else {
        importances.into_iter().map(|(k, v)| (k, v / total)).collect()
    }
}

fn default_target(t: &PersistedTrial) -> f64 {
    match &t.state_values {
        TrialStateValues::Complete(values) => {
            assert_eq!(values.len(), 1, "Specify the `target` function for multi-objective studies.");
            values[0]
        }
        _ => unreachable!("Only completed trials should be evaluated."),
    }
}

pub(crate) fn resolve_target(target: Option<&dyn Fn(&PersistedTrial) -> f64>)
    -> &dyn Fn(&PersistedTrial) -> f64
{
    target.unwrap_or(&default_target)
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