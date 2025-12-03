use std::collections::HashMap;

use crate::attr::Attrs;
use crate::distribution::Distribution;
use crate::study::{Direction, PersistedStudy};
use crate::study_cache::StudyCache;
use crate::trial::{PersistedTrial, TrialStateValues};
use crate::{Error, ErrorKind, Result};

pub trait Storage: Send + Sync {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy>;
    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial>;
    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()>;
    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state_values: TrialStateValues,
    ) -> Result<()>;
    // Design Note:
    // get_* methods take &mut self to allow in-place cache refresh in wrapper implementations
    // (e.g., CachedStorage). With &self it is impossible to safely update caches and return
    // references without relying on unsafe patterns.
    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>>;
    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy>;
    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>>;
    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial>;
    // Design Note:
    // Unlike the storage APIs in Optuna, the `set_study_attrs` and `set_trial_attrs` methods
    // are designed to receive multiple attributes for bulk insert operations.
    // Furthermore, the `user_attrs` and `system_attrs` are merged into a single HashMap,
    // which simplifies the implementation process for third-party storages.
    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()>;
    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()>;
    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>>;
}

#[derive(Default)]
pub struct InMemoryStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, Vec<PersistedTrial>>,
    study_caches: HashMap<u32, StudyCache>,
}
impl InMemoryStorage {
    pub fn new() -> InMemoryStorage {
        InMemoryStorage {
            studies: vec![],
            trials: HashMap::new(),
            study_caches: HashMap::new(),
        }
    }
}
fn get_trials_by_study_id(
    all_trials: &HashMap<u32, Vec<PersistedTrial>>,
    study_id: u32,
) -> Result<&Vec<PersistedTrial>> {
    let trials = all_trials
        .get(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
fn get_mut_trials_by_study_id(
    all_trials: &mut HashMap<u32, Vec<PersistedTrial>>,
    study_id: u32,
) -> Result<&mut Vec<PersistedTrial>> {
    let trials = all_trials
        .get_mut(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
impl Storage for InMemoryStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        if self.studies.iter().any(|s| s.name == study_name) {
            return Err(Error::new(ErrorKind::DuplicatedStudy));
        }

        let study_id = self.studies.len() as u32;
        self.studies.push(PersistedStudy::new(
            study_id,
            study_name.to_string(),
            directions,
        ));
        self.trials.insert(study_id, vec![]);
        Ok(&self.studies[study_id as usize])
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
        let number = trials.len() as u32;
        trials.push(PersistedTrial::new(study_id, number));
        Ok(&trials[number as usize])
    }

    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        check_trial_is_updatable(trial)?;

        // Check param distribution compatibility with previous trial(s).
        let study_distributions = &mut self
            .study_caches
            .entry(study_id)
            .or_default()
            .param_distribution;
        if let Some(study_distribution) = study_distributions.get(name) {
            study_distribution.check_compatibility(distribution)?;
        }
        study_distributions.insert(name.to_string(), distribution.clone());

        trial
            .distributions
            .insert(name.to_string(), distribution.clone());
        trial.internal_params.insert(name.to_string(), value);
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        check_trial_is_updatable(trial)?;
        trial.state_values = state_values;
        Ok(())
    }

    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>> {
        Ok(&self.studies)
    }

    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy> {
        let study = self
            .studies
            .get(study_id as usize)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        Ok(study)
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        get_trials_by_study_id(&self.trials, study_id)
    }

    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial> {
        let trial = get_trials_by_study_id(&self.trials, study_id)?
            .get(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        Ok(trial)
    }

    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()> {
        let study = self
            .studies
            .get_mut(study_id as usize)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        attrs.into_iter().for_each(|(key, value)| {
            study.attrs.insert(key, value);
        });
        Ok(())
    }

    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()> {
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        check_trial_is_updatable(trial)?;
        attrs.into_iter().for_each(|(key, value)| {
            trial.attrs.insert(key, value);
        });
        Ok(())
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        let study_cache = self
            .study_caches
            .entry(study_id)
            .or_default();
        let trials = get_trials_by_study_id(&self.trials, study_id)?;
        study_cache.update(trials);
        Ok(study_cache.get_joint_search_space())
    }
}

fn check_trial_is_updatable(trial: &PersistedTrial) -> Result<()> {
    if trial.is_finished() {
        return Err(Error::new(ErrorKind::TrialAlreadyFinished));
    }
    Ok(())
}
