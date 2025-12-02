use std::sync::Mutex;

use rusqlite::{params, Connection};
use rustuna_core::attr::{AttrKey, Attrs};
use rustuna_core::storage_cache::CachedStorageBackend;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::{Error, ErrorKind, Result};

pub struct SQLite3Storage {
    conn: Mutex<Connection>,
}

const SCHEMA_SQL: &str = include_str!("sqlite3_schema.sql");

impl SQLite3Storage {
    pub fn new(file_path: &str) -> Result<SQLite3Storage> {
        let conn = Connection::open(file_path).map_err(|_e| Error::new(ErrorKind::StorageError))?;
        Ok(SQLite3Storage {
            conn: Mutex::new(conn),
        })
    }

    pub fn create_database(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        Ok(())
    }
}

impl CachedStorageBackend for SQLite3Storage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<rustuna_core::study::Direction>,
    ) -> rustuna_core::Result<rustuna_core::study::PersistedStudy> {
        todo!()
    }

    fn create_new_trial(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<rustuna_core::trial::PersistedTrial> {
        todo!()
    }

    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: &str,
        distribution: &rustuna_core::distribution::Distribution,
        value: f64,
    ) -> rustuna_core::Result<()> {
        todo!()
    }

    fn set_trial_state_values(
        &mut self,
        _study_id: u32,
        _trial_number: u32,
        _state_values: rustuna_core::trial::TrialStateValues,
    ) -> rustuna_core::Result<()> {
        todo!()
    }

    fn get_studies(&mut self) -> rustuna_core::Result<Vec<rustuna_core::study::PersistedStudy>> {
        let guard = self.conn.lock().unwrap();

        let mut studies = Vec::new();
        let mut stmt = guard
            .prepare("SELECT study_id, study_name FROM studies ORDER BY study_id")
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        for row in rows {
            let (study_id, study_name) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;

            // Directions
            let mut directions_stmt = guard
                .prepare(
                    "SELECT direction FROM study_directions WHERE study_id = ? ORDER BY objective",
                )
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let directions_rows = directions_stmt
                .query_map(params![study_id], |row| row.get::<_, String>(0))
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let mut directions: Vec<Direction> = Vec::new();
            for d in directions_rows {
                let dir_str = d.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                let dir = match dir_str.as_str() {
                    "MINIMIZE" => Direction::Minimize,
                    "MAXIMIZE" => Direction::Maximize,
                    _ => return Err(Error::new(ErrorKind::StorageError)),
                };
                directions.push(dir);
            }

            // Attributes
            let mut attrs: Attrs = Attrs::new();

            let mut user_stmt = guard
                .prepare("SELECT key, value_json FROM study_user_attributes WHERE study_id = ?")
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let user_rows = user_stmt
                .query_map(params![study_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            for row in user_rows {
                let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                attrs.insert(AttrKey::User(key), value);
            }

            let mut system_stmt = guard
                .prepare("SELECT key, value_json FROM study_system_attributes WHERE study_id = ?")
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let system_rows = system_stmt
                .query_map(params![study_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            for row in system_rows {
                let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                attrs.insert(AttrKey::System(key), value);
            }

            let study = PersistedStudy::new_with_attrs(study_id, study_name, directions, attrs);
            studies.push(study);
        }
        Ok(studies)
    }

    fn get_study(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<rustuna_core::study::PersistedStudy> {
        let studies = self.get_studies()?;
        studies
            .into_iter()
            .find(|s| s.id == study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))
    }

    fn get_trial(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> rustuna_core::Result<rustuna_core::trial::PersistedTrial> {
        todo!()
    }

    fn set_study_attrs(
        &mut self,
        _study_id: u32,
        _attrs: rustuna_core::attr::Attrs,
    ) -> rustuna_core::Result<()> {
        todo!()
    }

    fn set_trial_attrs(
        &mut self,
        _study_id: u32,
        _trial_number: u32,
        _attrs: rustuna_core::attr::Attrs,
    ) -> rustuna_core::Result<()> {
        todo!()
    }

    fn get_trials_diff(
        &mut self,
        _study_id: u32,
        _included_numbers: &[u32],
        _trial_number_greater_than: i32,
    ) -> rustuna_core::Result<Vec<rustuna_core::trial::PersistedTrial>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::distribution::Distribution;
    use rustuna_core::study::Direction;

    fn init_storage() -> SQLite3Storage {
        let storage = SQLite3Storage::new(":memory:").unwrap();
        storage.create_database().unwrap();
        storage
    }
}
