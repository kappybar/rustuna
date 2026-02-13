use rustuna_core::distribution::Distribution;
use rustuna_core::study::Study;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use std::collections::HashMap;

pub struct ImportanceOptions<'a> {
    // NOTE(kAIto47802): Currently, the `param` argument is not implemented.
    // We plan to implement it when we support condPED-ANOVA:
    // - https://arxiv.org/abs/2601.20800
    pub target: Option<&'a dyn Fn(&PersistedTrial) -> f64>,
    pub normalize: bool,
}

impl<'a> Default for ImportanceOptions<'a> {
    fn default() -> Self {
        Self {
            target: None,
            normalize: true,
        }
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

pub fn get_param_importances(
    study: &Study,
    evaluator: &impl ImportanceEvaluator,
) -> Result<HashMap<String, f64>> {
    get_param_importances_with(study, evaluator, ImportanceOptions::default())
}

pub fn get_param_importances_with(
    study: &Study,
    evaluator: &impl ImportanceEvaluator,
    opts: ImportanceOptions<'_>,
) -> Result<HashMap<String, f64>> {
    let normalize = opts.normalize;
    let importances = evaluator.evaluate_with(study, opts)?;
    if normalize {
        Ok(normalize_importances(importances))
    } else {
        Ok(importances)
    }
}

pub trait ImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> Result<HashMap<String, f64>> {
        self.evaluate_with(study, ImportanceOptions::default())
    }
    fn evaluate_with(
        &self,
        study: &Study,
        opts: ImportanceOptions<'_>,
    ) -> Result<HashMap<String, f64>>;
}

fn normalize_importances(importances: HashMap<String, f64>) -> HashMap<String, f64> {
    let total = importances.values().sum::<f64>();
    if total == 0.0 {
        let n_params = importances.len() as f64;
        importances
            .into_keys()
            .map(|k| (k, 1.0 / n_params))
            .collect()
    } else {
        importances
            .into_iter()
            .map(|(k, v)| (k, v / total))
            .collect()
    }
}

fn default_target(t: &PersistedTrial) -> f64 {
    match &t.state_values {
        TrialStateValues::Complete(values) => values[0],
        _ => unreachable!("Only completed trials should be evaluated."),
    }
}

pub(crate) fn ensure_target_for_multi_objective(
    trials: &[PersistedTrial],
    target: Option<&dyn Fn(&PersistedTrial) -> f64>,
) -> Result<()> {
    let Some(first) = trials.first() else {
        return Ok(());
    };
    match &first.state_values {
        TrialStateValues::Complete(values) => {
            if target.is_some() || values.len() == 1 {
                Ok(())
            } else {
                Err(Error::with_reason(
                    ErrorKind::ImportanceEvaluatorError,
                    "Specify the `target` function for multi-objective studies.",
                ))
            }
        }
        _ => unreachable!("Only completed trials should be evaluated."),
    }
}

pub(crate) fn resolve_target(
    target: Option<&dyn Fn(&PersistedTrial) -> f64>,
) -> &dyn Fn(&PersistedTrial) -> f64 {
    target.unwrap_or(&default_target)
}

pub(crate) fn get_filtered_trials(
    study: &Study,
    target: &dyn Fn(&PersistedTrial) -> f64,
) -> Result<Vec<PersistedTrial>> {
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

pub(crate) fn get_intersection_search_space(
    trials: &[PersistedTrial],
) -> HashMap<String, Distribution> {
    let mut intersection_search_space = trials[0].distributions.clone();
    for trial in &trials[1..] {
        intersection_search_space
            .retain(|k, v| trial.distributions.get(k).is_some_and(|v2| v2 == v));
    }
    intersection_search_space
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::collections::HashSet;
    use rustuna_core::sampler::RandomSampler;
    use rustuna_core::{ErrorKind, Result};
    use crate::test_utils;
    use crate::ped_anova::PedAnovaImportanceEvaluator;
    use rustuna_core::study::{self, Direction};
    use rustuna_core::trial::PersistedTrial;
    use rustuna_core::storage::InMemoryStorage;


    #[test]
    fn test_error_multi_objective_wo_target() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = test_utils::get_study(42, 5, true, Direction::Minimize)?;
        for evaluator in evaluators {
            let err = get_param_importances(&study, &evaluator).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::ImportanceEvaluatorError));
        }
        Ok(())
    }

    #[test]
    fn test_evaluator_error_multi_objective_wo_target() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = test_utils::get_study(42, 5, true, Direction::Minimize)?;
        for evaluator in evaluators {
            let err = evaluator.evaluate(&study).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::ImportanceEvaluatorError));
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = test_utils::get_study(42, 20, false, Direction::Minimize)?;
        for evaluator in evaluators {
            for normalize in [true, false] {
                let importances = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new().normalize(normalize),
                )?;
                assert_eq!(!importances.len(), 6);
                if normalize {
                    assert!(importances.values().all(|v| (-1e-12..=1.0 + 1e-12).contains(v)));
                    assert!((importances.values().sum::<f64>() - 1.0).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_with_target() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = test_utils::get_study(42, 20, false, Direction::Minimize)?;
        let target = |t: &PersistedTrial| -> f64 {
            t.internal_params["x1"] + t.internal_params["x2"]
        };
        for evaluator in evaluators {
            for normalize in [true, false] {
                let importances = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new()
                        .with_target(&target)
                        .normalize(normalize),
                )?;
                assert_eq!(!importances.len(), 6);
                if normalize {
                    assert!(importances.values().all(|v| (-1e-12..=1.0 + 1e-12).contains(v)));
                    assert!((importances.values().sum::<f64>() - 1.0).abs() < 1e-12);
                }

                let importances_wo_target = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new().normalize(normalize),
                )?;
                assert_ne!(importances, importances_wo_target);
            }
        }
        Ok(())
    }

    #[test]
    fn test_evaluator_evaluate_with_target() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = test_utils::get_study(42, 20, false, Direction::Minimize)?;
        let target = |t: &PersistedTrial| -> f64 {
            t.internal_params["x1"] + t.internal_params["x2"]
        };
        for evaluator in evaluators {
            for normalize in [true, false] {
                let importances = evaluator.evaluate_with(
                    &study,
                    ImportanceOptions::new()
                        .with_target(&target)
                        .normalize(normalize),
                )?;
                assert_eq!(!importances.len(), 6);
                if normalize {
                    assert!(importances.values().all(|v| (-1e-12..=1.0 + 1e-12).contains(v)));
                    assert!((importances.values().sum::<f64>() - 1.0).abs() < 1e-12);
                }
                let importances_wo_target = evaluator.evaluate_with(
                    &study,
                    ImportanceOptions::new().normalize(normalize),
                )?;
                assert_ne!(importances, importances_wo_target);
            }
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_empty_study() -> Result<()> {
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        let study = study::create_study(
            "empty-study",
            InMemoryStorage::new(),
            vec![Direction::Minimize],
        )?;
        for evaluator in evaluators {
            let err = get_param_importances(&study, &evaluator).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::NoCompletedTrial));
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_empty_search_space() -> Result<()> {
        let mut study = study::create_study(
            "empty-search-space",
            InMemoryStorage::new(),
            vec![Direction::Minimize],
        )?;
        study.optimize(
            |mut t| {
                let x1 = t.suggest_float("x1", 0.0, 5.0)?;
                let x2 = t.suggest_float("x2", 1.0, 1.0)?;
                Ok(vec![x1 + x2])
            },
            Arc::new(Mutex::new(RandomSampler::new())),
            5,
        )?;
        let evaluators = vec![
            PedAnovaImportanceEvaluator::default(),
        ];
        for evaluator in evaluators {
            let importances = get_param_importances(&study, &evaluator)?;
            let keys = importances.keys().map(String::as_str).collect::<HashSet<_>>();
            let expected = HashSet::from(["x1", "x2"]);
            assert_eq!(keys, expected);
            assert!(importances["x1"] > 0.0);
            assert!(importances["x2"] == 0.0);
        }
        Ok(())
    }
}