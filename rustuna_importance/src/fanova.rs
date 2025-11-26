use fanova::{FanovaOptions, RandomForestOptions};

use rustuna_core::study::Study;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};

pub fn get_param_importance(study: &Study) -> Result<Vec<Vec<f64>>> {
    let guard = study
        .storage
        .read()
        .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
    // TODO(c-bata): Avoid to clone trials.
    let completed_trials: Vec<PersistedTrial> = guard
        .get_trials(study.id)?
        .clone()
        .into_iter()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(_)))
        .collect();
    drop(guard);
    if completed_trials.is_empty() {
        return Err(Error::new(ErrorKind::NoCompletedTrial));
    }

    let mut intersection_search_space = completed_trials[0].distributions.clone();
    for trial in completed_trials.iter() {
        for (key, value) in trial.distributions.iter() {
            // TODO(c-bata): Check distribution compatibility.
            if intersection_search_space.contains_key(key)
                && intersection_search_space[key] != *value
            {
                intersection_search_space.remove(key);
            }
        }
    }
    let param_names: Vec<String> = intersection_search_space.keys().cloned().collect();
    let mut features: Vec<Vec<f64>> = param_names.iter().map(|_| vec![]).collect();
    let mut values: Vec<Vec<f64>> = study.directions.iter().map(|_| vec![]).collect();
    for t in completed_trials.iter() {
        if let TrialStateValues::Complete(v) = &t.state_values {
            if v.len() != study.directions.len() {
                continue; // Invalid length of values
            }

            for param_index in 0..param_names.len() {
                let name: &str = &param_names[param_index];
                let f = t
                    .internal_params
                    .get(name)
                    .ok_or(Error::new(ErrorKind::Unexpected))?;
                features[param_index].push(*f);
            }

            for objective_id in 0..v.len() {
                values[objective_id].push(v[objective_id]);
            }
        }
    }

    let mut importances = vec![];
    for target in values.iter() {
        let features_vec = features.iter().map(|x| x.as_slice()).collect();
        let mut fanova = FanovaOptions::new()
            .random_forest(RandomForestOptions::new().seed(0))
            .fit(features_vec, target)
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let importance = (0..features.len())
            .map(|i| fanova.quantify_importance(&[i]).mean)
            .collect::<Vec<_>>();
        importances.push(importance);
    }

    Ok(importances)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rustuna_core::sampler::RandomSampler;
    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{create_study, Direction};

    use super::*;

    #[test]
    fn fanova() {
        let storage = InMemoryStorage::new();
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let mut study = create_study("dummy", storage, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let z = t.suggest_int("z", 0, 10)?;

                    let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                    Ok(vec![value])
                },
                sampler,
                100,
            )
            .unwrap();

        let importances = get_param_importance(&study).unwrap();
        assert_eq!(importances.len(), 1);
        assert_eq!(importances[0].len(), 3);
    }
}
