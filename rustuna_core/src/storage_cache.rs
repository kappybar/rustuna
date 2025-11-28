use std::collections::HashMap;

use crate::attr::Attrs;
use crate::distribution::Distribution;
use crate::study::{Direction, PersistedStudy};
use crate::study_cache::StudyCache;
use crate::trial::{PersistedTrial, TrialStateValues};
use crate::{Result};


pub trait CachedStorageBackend: Send + Sync {
    // Design Note:
    // This trait is intended for backends that return owned values (not references) so that
    // a wrapper (e.g., CachedStorage) can materialize in-memory caches and then hand out
    // references required by the Storage trait. This mirrors Optuna's _CachedStorage pattern:
    // backend focuses on persistence, wrapper handles caching and reference semantics.
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<PersistedStudy>;
    fn create_new_trial(&mut self, study_id: u32) -> Result<PersistedTrial>;
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
    fn get_studies(&self) -> Result<Vec<PersistedStudy>>;
    fn get_study(&self, study_id: u32) -> Result<PersistedStudy>;
    fn get_trials(&self, study_id: u32) -> Result<Vec<PersistedTrial>>;
    fn get_trial(&self, study_id: u32, trial_number: u32) -> Result<PersistedTrial>;
    // Design Note:
    // Unlike the storage APIs in Optuna, the `set_study_attrs` and `set_trial_attrs` methods
    // are designed to receive multiple attributes for bulk insert operations.
    // Furthermore, the `user_attrs` and `system_attrs` are merged into a single HashMap,
    // which simplifies the implementation process for third-party storages.
    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()>;
    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()>;
    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>>;
}


pub struct CachedStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, Vec<PersistedTrial>>,
    study_caches: HashMap<u32, StudyCache>,

    backend: Box<dyn CachedStorageBackend>,
}

impl CachedStorage {
    pub fn new(backend: Box<dyn CachedStorageBackend>) -> CachedStorage {
        CachedStorage {
            studies: Vec::new(),
            trials: HashMap::new(),
            study_caches: HashMap::new(),
            backend,
        }
    }
}

impl crate::storage::Storage for CachedStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        todo!()
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        todo!()
    }

    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        todo!()
    }

    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        todo!()
    }

    fn get_studies(&self) -> Result<&Vec<PersistedStudy>> {
        todo!()
    }

    fn get_study(&self, study_id: u32) -> Result<&PersistedStudy> {
        todo!()
    }

    fn get_trials(&self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        todo!()
    }

    fn get_trial(&self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial> {
        todo!()
    }

    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()> {
        todo!()
    }

    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()> {
        todo!()
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;

    struct DummyBackend {
        inner: crate::storage::InMemoryStorage,
    }

    impl DummyBackend {
        fn new() -> Self {
            DummyBackend {
                inner: crate::storage::InMemoryStorage::new(),
            }
        }
    }

    impl CachedStorageBackend for DummyBackend {
        fn create_new_study(
            &mut self,
            study_name: &str,
            directions: Vec<Direction>,
        ) -> Result<PersistedStudy> {
            let study = self.inner.create_new_study(study_name, directions)?.clone();
            Ok(study)
        }

        fn create_new_trial(&mut self, study_id: u32) -> Result<PersistedTrial> {
            let trial = self.inner.create_new_trial(study_id)?.clone();
            Ok(trial)
        }

        fn set_trial_param(
            &mut self,
            study_id: u32,
            trial_number: u32,
            name: &str,
            distribution: &Distribution,
            value: f64,
        ) -> Result<()> {
            self.inner
                .set_trial_param(study_id, trial_number, name, distribution, value)
        }

        fn set_trial_state_values(
            &mut self,
            study_id: u32,
            trial_number: u32,
            state_values: TrialStateValues,
        ) -> Result<()> {
            self.inner
                .set_trial_state_values(study_id, trial_number, state_values)
        }

        fn get_studies(&self) -> Result<Vec<PersistedStudy>> {
            Ok(self.inner.get_studies()?.clone())
        }

        fn get_study(&self, study_id: u32) -> Result<PersistedStudy> {
            Ok(self.inner.get_study(study_id)?.clone())
        }

        fn get_trials(&self, study_id: u32) -> Result<Vec<PersistedTrial>> {
            Ok(self.inner.get_trials(study_id)?.clone())
        }

        fn get_trial(&self, study_id: u32, trial_number: u32) -> Result<PersistedTrial> {
            Ok(self.inner.get_trial(study_id, trial_number)?.clone())
        }

        fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()> {
            self.inner.set_study_attrs(study_id, attrs)
        }

        fn set_trial_attrs(
            &mut self,
            study_id: u32,
            trial_number: u32,
            attrs: Attrs,
        ) -> Result<()> {
            self.inner.set_trial_attrs(study_id, trial_number, attrs)
        }

        fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
            self.inner.get_joint_search_space(study_id)
        }
    }
}
