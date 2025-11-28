use std::collections::HashMap;

use crate::attr::Attrs;
use crate::distribution::Distribution;
use crate::study::{Direction, PersistedStudy};
use crate::study_cache::StudyCache;
use crate::trial::{PersistedTrial, TrialStateValues};
use crate::{Error, ErrorKind, Result};

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
    fn get_studies(&mut self) -> Result<Vec<PersistedStudy>>;
    fn get_study(&mut self, study_id: u32) -> Result<PersistedStudy>;
    fn get_trials(&mut self, study_id: u32) -> Result<Vec<PersistedTrial>>;
    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<PersistedTrial>;
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
        let study = self.backend.create_new_study(study_name, directions)?;
        let study_id = study.id;
        self.studies.push(study);
        self.trials.insert(study_id, vec![]);
        self.study_caches.insert(study_id, StudyCache::new());
        Ok(self.studies.last().unwrap())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trial = self.backend.create_new_trial(study_id)?;
        let trials = self.trials.entry(study_id).or_insert_with(Vec::new);
        trials.push(trial);
        let trial_ref = trials.last().unwrap();

        let study_cache = self
            .study_caches
            .entry(study_id)
            .or_insert_with(StudyCache::new);
        study_cache.update(trials);
        Ok(trial_ref)
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

    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>> {
        let loaded = self.backend.get_studies()?;
        self.studies = loaded;
        Ok(&self.studies)
    }

    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy> {
        let loaded = self.backend.get_studies()?;
        self.studies = loaded;
        let study = self
            .studies
            .iter()
            .find(|s| s.id == study_id)
            .ok_or_else(|| crate::Error::new(crate::ErrorKind::StudyNotFound))?;
        Ok(study)
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        if !self.trials.contains_key(&study_id) {
            let loaded = self.backend.get_trials(study_id)?;
            self.trials.insert(study_id, loaded);
        }
        self.trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))
    }

    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial> {
        let trials = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get(trial_number as usize)
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        Ok(trial)
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
    use crate::ErrorKind;

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

        fn get_studies(&mut self) -> Result<Vec<PersistedStudy>> {
            Ok(self.inner.get_studies()?.clone())
        }

        fn get_study(&mut self, study_id: u32) -> Result<PersistedStudy> {
            Ok(self.inner.get_study(study_id)?.clone())
        }

        fn get_trials(&mut self, study_id: u32) -> Result<Vec<PersistedTrial>> {
            Ok(self.inner.get_trials(study_id)?.clone())
        }

        fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<PersistedTrial> {
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

        fn get_joint_search_space(
            &mut self,
            study_id: u32,
        ) -> Result<HashMap<String, Distribution>> {
            self.inner.get_joint_search_space(study_id)
        }
    }

    #[test]
    fn create_new_study_updates_cache() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let (study_id, name, directions) = {
            let study = storage
                .create_new_study("example", vec![Direction::Minimize])
                .unwrap();
            (study.id, study.name.clone(), study.directions.clone())
        };
        assert_eq!(name, "example");
        assert_eq!(directions, vec![Direction::Minimize]);
        assert_eq!(storage.studies.len(), 1);
        assert!(storage.trials.get(&study_id).is_some());
    }

    #[test]
    fn create_new_study_rejects_duplicate() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        storage
            .create_new_study("example", vec![Direction::Minimize])
            .unwrap();
        let res = storage.create_new_study("example", vec![Direction::Minimize]);
        match res {
            Err(e) => assert!(matches!(e.kind, ErrorKind::DuplicatedStudy)),
            Ok(_) => panic!("Expected duplicate study error"),
        }
    }

    #[test]
    fn get_study_and_get_studies_use_cache() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        storage
            .create_new_study("s1", vec![Direction::Minimize])
            .unwrap();
        storage
            .create_new_study("s2", vec![Direction::Maximize])
            .unwrap();

        let all = storage.get_studies().unwrap();
        assert_eq!(all.len(), 2);

        let s1 = storage.get_study(0).unwrap();
        assert_eq!(s1.name, "s1");
        let s2 = storage.get_study(1).unwrap();
        assert_eq!(s2.name, "s2");
    }

    #[test]
    fn create_new_trial_appends_cache() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study = storage
            .create_new_study("s", vec![Direction::Minimize])
            .unwrap()
            .id;
        let t0_num = storage.create_new_trial(study).unwrap().number;
        let t1_num = storage.create_new_trial(study).unwrap().number;
        assert_eq!(t0_num, 0);
        assert_eq!(t1_num, 1);
        let trials = storage.trials.get(&study).unwrap();
        assert_eq!(trials.len(), 2);
    }

    #[test]
    fn get_trials_and_get_trial_return_cached_refs() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage
            .create_new_study("s", vec![Direction::Minimize])
            .unwrap()
            .id;
        storage.create_new_trial(study_id).unwrap();
        storage.create_new_trial(study_id).unwrap();

        let trials = storage.get_trials(study_id).unwrap();
        assert_eq!(trials.len(), 2);
        let t0 = storage.get_trial(study_id, 0).unwrap();
        assert_eq!(t0.number, 0);
        let t1 = storage.get_trial(study_id, 1).unwrap();
        assert_eq!(t1.number, 1);
    }

    #[test]
    fn get_trials_loads_from_backend_when_empty() {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage
            .create_new_study("s", vec![Direction::Minimize])
            .unwrap()
            .id;
        storage.create_new_trial(study_id).unwrap();

        storage.trials.clear();
        let trials = storage.get_trials(study_id).unwrap();
        assert_eq!(trials.len(), 1);
    }

    #[test]
    fn get_studies_refreshes_from_backend_every_time() {
        let mut backend = DummyBackend::new();
        let study = backend
            .create_new_study("s", vec![Direction::Minimize])
            .unwrap();
        let mut storage = CachedStorage::new(Box::new(backend));

        // First load
        let studies = storage.get_studies().unwrap();
        assert_eq!(studies.len(), 1);

        // Add another study directly to backend and ensure get_studies reflects it.
        let _ = storage
            .backend
            .create_new_study("s2", vec![Direction::Maximize])
            .unwrap();
        let studies = storage.get_studies().unwrap();
        assert_eq!(studies.len(), 2);
        assert!(studies.iter().any(|s| s.name == study.name));
        assert!(studies.iter().any(|s| s.name == "s2"));
    }
}
