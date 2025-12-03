use std::collections::HashMap;

use rustuna_core::attr::Attrs;
use rustuna_core::distribution::Distribution;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::study_cache::StudyCache;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};

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
    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<PersistedTrial>;
    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()>;
    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()>;

    // Return trials that need refreshing: unfinished trials in `included_numbers`
    // and trials with trial_number greater than `trial_number_greater_than`.
    fn get_trials_diff(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
    ) -> Result<Vec<PersistedTrial>>;
}

pub struct CachedStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, HashMap<u32, PersistedTrial>>,
    study_caches: HashMap<u32, StudyCache>,
    unfinished_trials: HashMap<u32, Vec<u32>>,
    last_finished_trial_number: HashMap<u32, i32>,
    trials_sorted_buffer: Vec<PersistedTrial>,

    backend: Box<dyn CachedStorageBackend>,
}

impl CachedStorage {
    pub fn new(backend: Box<dyn CachedStorageBackend>) -> CachedStorage {
        CachedStorage {
            studies: Vec::new(),
            trials: HashMap::new(),
            study_caches: HashMap::new(),
            unfinished_trials: HashMap::new(),
            last_finished_trial_number: HashMap::new(),
            trials_sorted_buffer: Vec::new(),
            backend,
        }
    }

    fn refresh_trials(&mut self, study_id: u32) -> Result<()> {
        let unfinished = self
            .unfinished_trials
            .get(&study_id)
            .cloned()
            .unwrap_or_default();
        let last_finished = self
            .last_finished_trial_number
            .get(&study_id)
            .copied()
            .unwrap_or(-1);
        let loaded = self
            .backend
            .get_trials_diff(study_id, &unfinished, last_finished)?;

        if loaded.is_empty() {
            return Ok(());
        }

        let trials = self.trials.entry(study_id).or_default();
        for trial in loaded {
            trials.insert(trial.number, trial);
        }

        let study_cache = self.study_caches.entry(study_id).or_default();
        let mut trials_vec: Vec<_> = trials.values().cloned().collect();
        trials_vec.sort_by_key(|t| t.number);
        study_cache.update(&trials_vec);

        let mut unfinished_next = vec![];
        let mut last_finished_next = last_finished;
        for trial in trials.values() {
            if trial.is_finished() {
                last_finished_next = last_finished_next.max(trial.number as i32);
            } else {
                unfinished_next.push(trial.number);
            }
        }
        self.unfinished_trials.insert(study_id, unfinished_next);
        self.last_finished_trial_number
            .insert(study_id, last_finished_next);
        Ok(())
    }
}

impl rustuna_core::storage::Storage for CachedStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        let study = self.backend.create_new_study(study_name, directions)?;
        let study_id = study.id;
        self.studies.push(study);
        self.trials.insert(study_id, HashMap::new());
        self.study_caches.insert(study_id, StudyCache::new());
        self.unfinished_trials.insert(study_id, vec![]);
        self.last_finished_trial_number.insert(study_id, -1);
        Ok(self.studies.last().unwrap())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trial = self.backend.create_new_trial(study_id)?;
        let trials = self.trials.entry(study_id).or_default();
        let number = trial.number;
        trials.insert(number, trial);
        let trial_ref = trials.get(&number).unwrap();

        let study_cache = self.study_caches.entry(study_id).or_default();
        let mut trials_vec: Vec<_> = trials.values().cloned().collect();
        trials_vec.sort_by_key(|t| t.number);
        study_cache.update(&trials_vec);
        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_ref.number);
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
        self.backend
            .set_trial_param(study_id, trial_number, name, distribution, value)?;
        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_number);
        self.refresh_trials(study_id)?;

        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(&trial_number)
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        trial
            .distributions
            .insert(name.to_string(), distribution.clone());
        trial.internal_params.insert(name.to_string(), value);

        let mut trials_vec: Vec<_> = trials.values().cloned().collect();
        trials_vec.sort_by_key(|t| t.number);
        self.study_caches
            .entry(study_id)
            .or_default()
            .update(&trials_vec);
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        self.backend
            .set_trial_state_values(study_id, trial_number, state_values.clone())?;

        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_number);
        self.refresh_trials(study_id)?;

        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(&trial_number)
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        trial.state_values = state_values;

        let mut trials_vec: Vec<_> = trials.values().cloned().collect();
        trials_vec.sort_by_key(|t| t.number);
        self.study_caches
            .entry(study_id)
            .or_default()
            .update(&trials_vec);
        Ok(())
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
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        Ok(study)
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        self.refresh_trials(study_id)?;
        let trials_map = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let mut trials_vec: Vec<_> = trials_map.values().cloned().collect();
        trials_vec.sort_by_key(|t| t.number);
        self.trials_sorted_buffer.clear();
        self.trials_sorted_buffer.extend(trials_vec);
        self.study_caches
            .entry(study_id)
            .or_default()
            .update(&self.trials_sorted_buffer);
        Ok(&self.trials_sorted_buffer)
    }

    fn get_trial(&mut self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial> {
        let trials = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get(&trial_number)
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        Ok(trial)
    }

    fn set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> Result<()> {
        self.backend.set_study_attrs(study_id, attrs.clone())?;
        self.studies = self.backend.get_studies()?;
        let study = self
            .studies
            .iter_mut()
            .find(|s| s.id == study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        for (k, v) in attrs {
            study.attrs.insert(k, v);
        }
        Ok(())
    }

    fn set_trial_attrs(&mut self, study_id: u32, trial_number: u32, attrs: Attrs) -> Result<()> {
        self.backend
            .set_trial_attrs(study_id, trial_number, attrs.clone())?;
        self.refresh_trials(study_id)?;
        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(&trial_number)
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        for (k, v) in attrs {
            trial.attrs.insert(k, v);
        }
        Ok(())
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        let trials_vec = {
            let trials = self.get_trials(study_id)?;
            let mut v = trials.clone();
            v.sort_by_key(|t| t.number);
            v
        };
        let cache = self.study_caches.entry(study_id).or_default();
        cache.update(&trials_vec);
        Ok(cache.get_joint_search_space())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::attr::AttrKey;
    use rustuna_core::storage::Storage;
    use rustuna_core::ErrorKind;

    struct DummyBackend {
        inner: rustuna_core::storage::InMemoryStorage,
    }

    impl DummyBackend {
        fn new() -> Self {
            DummyBackend {
                inner: rustuna_core::storage::InMemoryStorage::new(),
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

        fn get_trials_diff(
            &mut self,
            study_id: u32,
            included_numbers: &[u32],
            trial_number_greater_than: i32,
        ) -> Result<Vec<PersistedTrial>> {
            let all = self.inner.get_trials(study_id)?.clone();
            let mut trials = Vec::new();
            for t in all {
                if included_numbers.contains(&t.number)
                    || (t.number as i32) > trial_number_greater_than
                {
                    trials.push(t);
                }
            }
            Ok(trials)
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
    }

    #[test]
    fn create_new_study_updates_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let (study_id, name, directions) = {
            let study = storage.create_new_study("example", vec![Direction::Minimize])?;
            (study.id, study.name.clone(), study.directions.clone())
        };
        assert_eq!(name, "example");
        assert_eq!(directions, vec![Direction::Minimize]);
        assert_eq!(storage.studies.len(), 1);
        assert!(storage.trials.get(&study_id).is_some());
        Ok(())
    }

    #[test]
    fn create_new_study_rejects_duplicate() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        storage.create_new_study("example", vec![Direction::Minimize])?;
        let res = storage.create_new_study("example", vec![Direction::Minimize]);
        match res {
            Err(e) => assert!(matches!(e.kind, ErrorKind::DuplicatedStudy)),
            Ok(_) => panic!("Expected duplicate study error"),
        }
        Ok(())
    }

    #[test]
    fn get_study_and_get_studies_use_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        storage.create_new_study("s1", vec![Direction::Minimize])?;
        storage.create_new_study("s2", vec![Direction::Maximize])?;

        let all = storage.get_studies()?;
        assert_eq!(all.len(), 2);

        let s1 = storage.get_study(0)?;
        assert_eq!(s1.name, "s1");
        let s2 = storage.get_study(1)?;
        assert_eq!(s2.name, "s2");
        Ok(())
    }

    #[test]
    fn create_new_trial_appends_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let t0_num = storage.create_new_trial(study)?.number;
        let t1_num = storage.create_new_trial(study)?.number;
        assert_eq!(t0_num, 0);
        assert_eq!(t1_num, 1);
        let trials = storage.trials.get(&study).unwrap();
        assert_eq!(trials.len(), 2);
        Ok(())
    }

    #[test]
    fn get_trials_and_get_trial_return_cached_refs() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;
        storage.create_new_trial(study_id)?;

        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 2);
        let t0 = storage.get_trial(study_id, 0)?;
        assert_eq!(t0.number, 0);
        let t1 = storage.get_trial(study_id, 1)?;
        assert_eq!(t1.number, 1);
        Ok(())
    }

    #[test]
    fn get_trials_loads_from_backend_when_empty() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        storage.trials.clear();
        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 1);
        Ok(())
    }

    #[test]
    fn get_studies_refreshes_from_backend_every_time() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study = backend.create_new_study("s", vec![Direction::Minimize])?;
        let mut storage = CachedStorage::new(Box::new(backend));

        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 1);

        storage
            .backend
            .create_new_study("s2", vec![Direction::Maximize])?;
        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 2);
        assert!(studies.iter().any(|s| s.name == study.name));
        assert!(studies.iter().any(|s| s.name == "s2"));
        Ok(())
    }

    #[test]
    fn get_trials_refreshes_when_backend_updates() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        backend.create_new_trial(study_id)?;

        let mut storage = CachedStorage::new(Box::new(backend));
        let trials1 = storage.get_trials(study_id)?;
        assert_eq!(trials1.len(), 1);

        storage.backend.create_new_trial(study_id)?;
        let trials2 = storage.get_trials(study_id)?;
        assert_eq!(trials2.len(), 2);
        Ok(())
    }

    #[test]
    fn set_trial_state_values_updates_cache() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        backend.create_new_trial(study_id)?;

        let mut storage = CachedStorage::new(Box::new(backend));
        storage.set_trial_state_values(study_id, 0, TrialStateValues::Complete(vec![1.0]))?;
        let trial = storage.get_trial(study_id, 0)?;
        assert!(matches!(trial.state_values, TrialStateValues::Complete(_)));
        Ok(())
    }

    #[test]
    fn get_joint_search_space_uses_cache_update() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;

        let dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        storage.create_new_trial(study_id)?;
        storage.set_trial_param(study_id, 0, "x", &dist, 0.5)?;
        storage.set_trial_state_values(study_id, 0, TrialStateValues::Complete(vec![0.0]))?;

        let search_space = storage.get_joint_search_space(study_id)?;
        assert!(search_space.contains_key("x"));
        Ok(())
    }

    #[test]
    fn set_study_and_trial_attrs_update_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()));
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        let mut s_attrs = Attrs::new();
        s_attrs.insert(AttrKey::User("foo".to_string()), "bar".to_string());
        storage.set_study_attrs(study_id, s_attrs)?;
        let study = storage.get_study(study_id)?;
        assert_eq!(
            study.attrs.get(&AttrKey::User("foo".to_string())).unwrap(),
            "bar"
        );

        let mut t_attrs = Attrs::new();
        t_attrs.insert(AttrKey::System("key".to_string()), "val".to_string());
        storage.set_trial_attrs(study_id, 0, t_attrs)?;
        let trial = storage.get_trial(study_id, 0)?;
        assert_eq!(
            trial
                .attrs
                .get(&AttrKey::System("key".to_string()))
                .unwrap(),
            "val"
        );
        Ok(())
    }

    #[test]
    fn set_trial_param_updates_cache_and_refreshes() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        backend.create_new_trial(study_id)?;

        let mut storage = CachedStorage::new(Box::new(backend));
        let dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        storage.set_trial_param(study_id, 0, "x", &dist, 0.5)?;

        let trial = storage.get_trial(study_id, 0)?;
        assert_eq!(trial.internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trial.distributions.get("x"),
            Some(&Distribution::Float {
                low: 0.0,
                high: 1.0,
                step: None,
                log: false
            })
        );
        Ok(())
    }
}
