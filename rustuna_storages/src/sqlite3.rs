use std::collections::HashMap;
use std::sync::Mutex;

use crate::cache::CachedStorageBackend;
use rusqlite::{params, Connection, OptionalExtension};
use rustuna_core::attr::{AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use serde_json::{json, Number, Value};

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

    fn validate_study_id(&self, study_id: u32) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let study_exists: Option<u32> = guard
            .query_row(
                "SELECT study_id FROM studies WHERE study_id = ?",
                params![study_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        if study_exists.is_none() {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }
        drop(guard);
        Ok(())
    }
}

impl CachedStorageBackend for SQLite3Storage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<rustuna_core::study::Direction>,
    ) -> rustuna_core::Result<rustuna_core::study::PersistedStudy> {
        let guard = self.conn.lock().unwrap();

        let existing: Option<u32> = guard
            .query_row(
                "SELECT study_id FROM studies WHERE study_name = ?",
                params![study_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        if existing.is_some() {
            return Err(Error::new(ErrorKind::DuplicatedStudy));
        }

        guard
            .execute(
                "INSERT INTO studies (study_name) VALUES (?)",
                params![study_name],
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        let study_id = guard.last_insert_rowid() as u32;

        for (objective, direction) in directions.iter().enumerate() {
            let direction_str = match direction {
                Direction::Minimize => "MINIMIZE",
                Direction::Maximize => "MAXIMIZE",
            };

            guard.execute(
                "INSERT INTO study_directions (direction, study_id, objective) VALUES (?, ?, ?)",
                params![direction_str, study_id, objective as u32],
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        }
        drop(guard);

        let persisted_study = PersistedStudy::new(study_id, study_name.to_string(), directions);
        Ok(persisted_study)
    }

    fn create_new_trial(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<rustuna_core::trial::PersistedTrial> {
        self.validate_study_id(study_id)?;

        let guard = self.conn.lock().unwrap();
        guard
            .execute(
                "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) \
             VALUES (NULL, ?, ?, CURRENT_TIMESTAMP, NULL)",
                params![study_id, "RUNNING"],
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        let trial_id = guard.last_insert_rowid() as u32;
        let number: u32 = guard
            .query_row(
                "SELECT COUNT(trial_id) FROM trials WHERE study_id = ? AND trial_id < ?",
                params![study_id, trial_id],
                |row| row.get(0),
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        guard
            .execute(
                "UPDATE trials SET number = ? WHERE trial_id = ?",
                params![number, trial_id],
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        Ok(PersistedTrial::new(study_id, number))
    }

    fn set_trial_param(
        &mut self,
        study_id: u32,
        trial_number: u32,
        name: &str,
        distribution: &rustuna_core::distribution::Distribution,
        value: f64,
    ) -> rustuna_core::Result<()> {
        let guard = self.conn.lock().unwrap();
        let trial_id: Option<u32> = guard
            .query_row(
                "SELECT trial_id FROM trials WHERE study_id = ? AND number = ?",
                params![study_id, trial_number],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let trial_id = trial_id.ok_or(Error::new(ErrorKind::TrialNotFound))?;

        let distribution_json = distribution_to_json(distribution, None);
        guard
            .execute(
                "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT(trial_id, param_name) DO UPDATE SET \
                 param_value=excluded.param_value, distribution_json=excluded.distribution_json",
                params![trial_id, name, value, distribution_json],
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        study_id: u32,
        trial_number: u32,
        state_values: rustuna_core::trial::TrialStateValues,
    ) -> rustuna_core::Result<()> {
        let guard = self.conn.lock().unwrap();
        let trial_id: Option<u32> = guard
            .query_row(
                "SELECT trial_id FROM trials WHERE study_id = ? AND number = ?",
                params![study_id, trial_number],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let trial_id = trial_id.ok_or(Error::new(ErrorKind::TrialNotFound))?;

        match &state_values {
            TrialStateValues::Complete(values) => {
                guard
                    .execute(
                        "UPDATE trials SET state = ?, datetime_complete = CURRENT_TIMESTAMP WHERE trial_id = ?",
                        params!["COMPLETE", trial_id],
                    )
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;

                if !values.is_empty() {
                    let placeholders = values
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("({trial_id}, {i}, ?, 'FINITE')"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let sql = format!(
                        "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES {placeholders} \
                         ON CONFLICT(trial_id, objective) DO UPDATE SET value=excluded.value, value_type=excluded.value_type"
                    );
                    let params: Vec<&dyn rusqlite::ToSql> =
                        values.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
                    guard
                        .execute(&sql, params.as_slice())
                        .map_err(|_e| Error::new(ErrorKind::StorageError))?;
                }
            }
            TrialStateValues::Pruned => {
                guard
                    .execute(
                        "UPDATE trials SET state = ?, datetime_complete = CURRENT_TIMESTAMP WHERE trial_id = ?",
                        params!["PRUNED", trial_id],
                    )
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            }
            TrialStateValues::Fail => {
                guard
                    .execute(
                        "UPDATE trials SET state = ?, datetime_complete = CURRENT_TIMESTAMP WHERE trial_id = ?",
                        params!["FAIL", trial_id],
                    )
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            }
            TrialStateValues::Running => {
                guard
                    .execute(
                        "UPDATE trials SET state = ?, datetime_complete = NULL WHERE trial_id = ?",
                        params!["RUNNING", trial_id],
                    )
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            }
            TrialStateValues::Waiting => {
                guard
                    .execute(
                        "UPDATE trials SET state = ?, datetime_complete = NULL WHERE trial_id = ?",
                        params!["WAITING", trial_id],
                    )
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            }
        }

        Ok(())
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
        let guard = self.conn.lock().unwrap();

        // Query to trials table .
        let trial_row: Option<(u32, String)> = guard
            .query_row(
                "SELECT trial_id, state FROM trials WHERE study_id = ? AND number = ?",
                params![study_id, trial_number],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let (trial_id, state_str) = trial_row.ok_or(Error::new(ErrorKind::TrialNotFound))?;
        let state_values = match state_str.as_str() {
            "RUNNING" | "WAITING" => TrialStateValues::Running,
            "PRUNED" => TrialStateValues::Pruned,
            "FAIL" => TrialStateValues::Fail,
            "COMPLETE" => {
                // Query to trial_values table.
                let mut stmt = guard
                    .prepare("SELECT value FROM trial_values WHERE trial_id = ? ORDER BY objective")
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
                let values = stmt
                    .query_map(params![trial_id], |row| row.get(0))
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?
                    .collect::<std::result::Result<Vec<f64>, _>>()
                    .map_err(|_e| Error::new(ErrorKind::StorageError))?;
                TrialStateValues::Complete(values)
            }
            _ => return Err(Error::new(ErrorKind::StorageError)),
        };

        // Query to trial_params table.
        let mut distributions = HashMap::new();
        let mut internal_params = HashMap::new();
        let mut stmt = guard
            .prepare(
                "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?",
            )
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let param_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        for row in param_rows {
            let (name, value, distribution_json) =
                row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let (distribution, _labels) = json_to_distribution(&distribution_json)?;
            distributions.insert(name.clone(), distribution);
            internal_params.insert(name, value);
        }

        // User attributes
        let mut attrs: Attrs = Attrs::new();
        let mut stmt = guard
            .prepare("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?")
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let user_attr_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        for row in user_attr_rows {
            let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
            attrs.insert(AttrKey::User(key), value);
        }

        // System attributes
        let mut stmt = guard
            .prepare("SELECT key, value_json FROM trial_system_attributes WHERE trial_id = ?")
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let system_attr_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        for row in system_attr_rows {
            let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
            attrs.insert(AttrKey::System(key), value);
        }

        // TODO(c-bata): Populate intermediate values into system attrs if needed.
        let mut trial = PersistedTrial::new(study_id, trial_number);
        trial.state_values = state_values;
        trial.internal_params = internal_params;
        trial.distributions = distributions;
        trial.attrs = attrs;
        Ok(trial)
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: rustuna_core::attr::Attrs,
    ) -> rustuna_core::Result<()> {
        self.validate_study_id(study_id)?;

        let mut user_attrs = Vec::new();
        let mut system_attrs = Vec::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(key_str) => user_attrs.push((key_str, value)),
                AttrKey::System(key_str) => system_attrs.push((key_str, value)),
            }
        }

        let guard = self.conn.lock().unwrap();
        if !user_attrs.is_empty() {
            let placeholders = user_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "INSERT INTO study_user_attributes (study_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(study_id, key) DO UPDATE SET value_json=excluded.value_json"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &user_attrs {
                params.push(&study_id);
                params.push(key);
                params.push(value);
            }
            guard
                .execute(&sql, params.as_slice())
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        }

        if !system_attrs.is_empty() {
            let placeholders = system_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "INSERT INTO study_system_attributes (study_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(study_id, key) DO UPDATE SET value_json=excluded.value_json"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &system_attrs {
                params.push(&study_id);
                params.push(key);
                params.push(value);
            }
            guard
                .execute(&sql, params.as_slice())
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        }

        Ok(())
    }

    fn set_trial_attrs(
        &mut self,
        study_id: u32,
        trial_number: u32,
        attrs: rustuna_core::attr::Attrs,
    ) -> rustuna_core::Result<()> {
        let mut user_attrs = Vec::new();
        let mut system_attrs = Vec::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(key_str) => user_attrs.push((key_str, value)),
                AttrKey::System(key_str) => system_attrs.push((key_str, value)),
            }
        }

        let guard = self.conn.lock().unwrap();
        let trial_id: Option<u32> = guard
            .query_row(
                "SELECT trial_id FROM trials WHERE study_id = ? AND number = ?",
                params![study_id, trial_number],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        let trial_id = trial_id.ok_or(Error::new(ErrorKind::TrialNotFound))?;

        if !user_attrs.is_empty() {
            let placeholders = user_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(trial_id, key) DO UPDATE SET value_json=excluded.value_json"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &user_attrs {
                params.push(&trial_id);
                params.push(key);
                params.push(value);
            }
            guard
                .execute(&sql, params.as_slice())
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        }

        if !system_attrs.is_empty() {
            let placeholders = system_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "INSERT INTO trial_system_attributes (trial_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(trial_id, key) DO UPDATE SET value_json=excluded.value_json"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &system_attrs {
                params.push(&trial_id);
                params.push(key);
                params.push(value);
            }
            guard
                .execute(&sql, params.as_slice())
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
        }

        Ok(())
    }

    fn get_trials_diff(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
    ) -> rustuna_core::Result<Vec<rustuna_core::trial::PersistedTrial>> {
        let guard = self.conn.lock().unwrap();

        // Build SQL query with filters
        let mut sql = String::from("SELECT trial_id, number, state FROM trials WHERE study_id = ?");
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(study_id)];

        // Filter by trial_number_greater_than
        if trial_number_greater_than >= 0 {
            sql.push_str(" AND number > ?");
            params.push(Box::new(trial_number_greater_than));
        }

        // Filter by included_numbers if provided
        if !included_numbers.is_empty() {
            let placeholders = included_numbers
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            sql.push_str(&format!(" OR number IN ({placeholders})"));
            for &num in included_numbers {
                params.push(Box::new(num));
            }
        }

        sql.push_str(" ORDER BY trial_id");

        let mut stmt = guard
            .prepare(&sql)
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let trial_rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, u32>(0)?,
                    row.get::<_, u32>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|_e| Error::new(ErrorKind::StorageError))?;

        let mut trials = Vec::new();
        for row in trial_rows {
            let (trial_id, number, state_str) =
                row.map_err(|_e| Error::new(ErrorKind::StorageError))?;

            // Parse state and get values if COMPLETE
            let state_values = match state_str.as_str() {
                "RUNNING" | "WAITING" => TrialStateValues::Running,
                "PRUNED" => TrialStateValues::Pruned,
                "FAIL" => TrialStateValues::Fail,
                "COMPLETE" => {
                    let mut values_stmt = guard
                        .prepare(
                            "SELECT value FROM trial_values WHERE trial_id = ? ORDER BY objective",
                        )
                        .map_err(|_e| Error::new(ErrorKind::StorageError))?;
                    let values = values_stmt
                        .query_map(params![trial_id], |row| row.get(0))
                        .map_err(|_e| Error::new(ErrorKind::StorageError))?
                        .collect::<std::result::Result<Vec<f64>, _>>()
                        .map_err(|_e| Error::new(ErrorKind::StorageError))?;
                    TrialStateValues::Complete(values)
                }
                _ => return Err(Error::new(ErrorKind::StorageError)),
            };

            // Get distributions and params
            let mut distributions = HashMap::new();
            let mut internal_params = HashMap::new();
            let mut params_stmt = guard
                .prepare("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?")
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let param_rows = params_stmt
                .query_map(params![trial_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            for row in param_rows {
                let (name, value, distribution_json) =
                    row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                let (distribution, _labels) = json_to_distribution(&distribution_json)?;
                distributions.insert(name.clone(), distribution);
                internal_params.insert(name, value);
            }

            // Get user attributes
            let mut attrs: Attrs = Attrs::new();
            let mut user_attrs_stmt = guard
                .prepare("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?")
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let user_attr_rows = user_attrs_stmt
                .query_map(params![trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            for row in user_attr_rows {
                let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                attrs.insert(AttrKey::User(key), value);
            }

            // Get system attributes
            let mut system_attrs_stmt = guard
                .prepare("SELECT key, value_json FROM trial_system_attributes WHERE trial_id = ?")
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            let system_attr_rows = system_attrs_stmt
                .query_map(params![trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|_e| Error::new(ErrorKind::StorageError))?;
            for row in system_attr_rows {
                let (key, value) = row.map_err(|_e| Error::new(ErrorKind::StorageError))?;
                attrs.insert(AttrKey::System(key), value);
            }

            let mut trial = PersistedTrial::new(study_id, number);
            trial.state_values = state_values;
            trial.internal_params = internal_params;
            trial.distributions = distributions;
            trial.attrs = attrs;
            trials.push(trial);
        }

        Ok(trials)
    }
}

fn distribution_to_json(distribution: &Distribution, labels: Option<&[CategoryLabel]>) -> String {
    let (name, attributes) = match distribution {
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => (
            "FloatDistribution",
            json!({
                "low": low,
                "high": high,
                "step": step,
                "log": log
            }),
        ),
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => (
            "IntDistribution",
            json!({
                "low": low,
                "high": high,
                "step": step,
                "log": log
            }),
        ),
        Distribution::Categorical { cardinality } => {
            let choices = labels
                .map(|ls| ls.iter().map(category_label_to_value).collect::<Vec<_>>())
                .unwrap_or_else(|| {
                    (0..*cardinality as u32)
                        .map(|i| serde_json::Value::Number(i.into()))
                        .collect::<Vec<_>>()
                });
            (
                "CategoricalDistribution",
                json!({
                    "choices": choices,
                }),
            )
        }
    };

    json!({
        "name": name,
        "attributes": attributes,
    })
    .to_string()
}

fn category_label_to_value(label: &CategoryLabel) -> Value {
    match label {
        CategoryLabel::Float(f) => Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        CategoryLabel::Int(i) => Value::Number(Number::from(*i)),
        CategoryLabel::String(s) => Value::String(s.clone()),
        CategoryLabel::Bool(b) => Value::Bool(*b),
        CategoryLabel::None => Value::Null,
    }
}

fn json_to_distribution(
    distribution_json: &str,
) -> Result<(Distribution, Option<Vec<CategoryLabel>>)> {
    let value: Value =
        serde_json::from_str(distribution_json).map_err(|_| Error::new(ErrorKind::StorageError))?;
    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
    let attributes = value
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::new(ErrorKind::StorageError))?;

    match name {
        "FloatDistribution" => {
            let low = attributes
                .get("low")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let step = match attributes.get("step") {
                Some(Value::Null) | None => None,
                Some(Value::Number(n)) => n.as_f64(),
                Some(Value::String(s)) => s.parse::<f64>().ok(),
                _ => None,
            };
            Ok((
                Distribution::Float {
                    low,
                    high,
                    step,
                    log,
                },
                None,
            ))
        }
        "IntDistribution" => {
            let low = attributes
                .get("low")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let step = match attributes.get("step") {
                Some(Value::Null) | None => None,
                Some(Value::Number(n)) => n.as_i64(),
                Some(Value::String(s)) => s.parse::<i64>().ok(),
                _ => None,
            };
            Ok((
                Distribution::Int {
                    low,
                    high,
                    step,
                    log,
                },
                None,
            ))
        }
        "CategoricalDistribution" => {
            let size = match attributes.get("size") {
                Some(v) => v.as_u64(),
                None => attributes
                    .get("choices")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() as u64),
            }
            .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
            let labels = attributes.get("choices").and_then(|arr| {
                arr.as_array().map(|vals| {
                    vals.iter()
                        .filter_map(value_to_category_label)
                        .collect::<Vec<_>>()
                })
            });
            Ok((
                Distribution::Categorical {
                    cardinality: size as usize,
                },
                labels,
            ))
        }
        _ => Err(Error::new(ErrorKind::StorageError)),
    }
}

fn value_to_category_label(v: &Value) -> Option<CategoryLabel> {
    match v {
        Value::Null => Some(CategoryLabel::None),
        Value::Bool(b) => Some(CategoryLabel::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(CategoryLabel::Int(i))
            } else {
                n.as_f64().map(CategoryLabel::Float)
            }
        }
        Value::String(s) => Some(CategoryLabel::String(s.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CachedStorage;
    use rustuna_core::sampler::RandomSampler;
    use rustuna_core::study::{create_study, Direction};
    use std::sync::Arc;

    fn init_storage() -> Result<SQLite3Storage> {
        let storage = SQLite3Storage::new(":memory:")?;
        storage.create_database()?;
        Ok(storage)
    }

    #[test]
    fn create_new_study_inserts_rows() -> Result<()> {
        let mut storage = init_storage()?;
        assert_eq!(storage.get_studies()?.len(), 0);

        let study =
            storage.create_new_study("example", vec![Direction::Minimize, Direction::Maximize])?;
        assert_eq!(study.name, "example");
        assert_eq!(
            study.directions,
            vec![Direction::Minimize, Direction::Maximize]
        );
        assert_eq!(storage.get_studies()?.len(), 1);
        Ok(())
    }

    #[test]
    fn create_new_study_rejects_duplicate_name() -> Result<()> {
        let mut storage = init_storage()?;
        storage.create_new_study("dup", vec![Direction::Minimize])?;
        let err = storage
            .create_new_study("dup", vec![Direction::Minimize])
            .err()
            .unwrap();
        assert!(matches!(err.kind, ErrorKind::DuplicatedStudy));
        Ok(())
    }

    #[test]
    fn create_new_trial_inserts_row() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state_values, TrialStateValues::Running);

        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.number, 1);
        Ok(())
    }

    #[test]
    fn set_trial_param() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        // FloatDistribution
        let float_dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        storage.set_trial_param(study_id, trial.number, "float", &float_dist, 0.5)?;

        // IntDistribution
        let int_dist = Distribution::Int {
            low: 0,
            high: 10,
            step: None,
            log: false,
        };
        storage.set_trial_param(study_id, trial.number, "int", &int_dist, 5.0)?;

        // CategoricalDistribution
        let categorical_dist = Distribution::Categorical { cardinality: 3 };
        storage.set_trial_param(study_id, trial.number, "cat", &categorical_dist, 1.0)?;

        // Check distributions
        let trial = storage.get_trial(study_id, trial.number)?;
        assert_eq!(trial.distributions.len(), 3);
        assert_eq!(trial.distributions["float"], float_dist);
        assert_eq!(trial.distributions["int"], int_dist);
        assert_eq!(trial.distributions["cat"], categorical_dist);

        // Check params
        assert_eq!(trial.internal_params.len(), 3);
        assert_eq!(trial.internal_params["float"], 0.5);
        assert_eq!(trial.internal_params["int"], 5.0);
        assert_eq!(trial.internal_params["cat"], 1.0);
        Ok(())
    }

    #[test]
    fn set_study_attrs() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let mut attrs = Attrs::new();
        attrs.insert(
            AttrKey::User("user_key".to_string()),
            "user_value".to_string(),
        );
        attrs.insert(
            AttrKey::System("system_key".to_string()),
            "system_value".to_string(),
        );

        storage.set_study_attrs(study_id, attrs)?;

        let study = storage.get_study(study_id)?;
        assert_eq!(study.attrs.len(), 2);
        assert_eq!(
            study.attrs.get(&AttrKey::User("user_key".to_string())),
            Some(&"user_value".to_string())
        );
        assert_eq!(
            study.attrs.get(&AttrKey::System("system_key".to_string())),
            Some(&"system_value".to_string())
        );
        Ok(())
    }

    #[test]
    fn set_trial_attrs() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        let mut attrs = Attrs::new();
        attrs.insert(
            AttrKey::User("trial_user_key".to_string()),
            "trial_user_value".to_string(),
        );
        attrs.insert(
            AttrKey::System("trial_system_key".to_string()),
            "trial_system_value".to_string(),
        );

        storage.set_trial_attrs(study_id, trial.number, attrs)?;

        let trial = storage.get_trial(study_id, trial.number)?;
        assert_eq!(trial.attrs.len(), 2);
        assert_eq!(
            trial
                .attrs
                .get(&AttrKey::User("trial_user_key".to_string())),
            Some(&"trial_user_value".to_string())
        );
        assert_eq!(
            trial
                .attrs
                .get(&AttrKey::System("trial_system_key".to_string())),
            Some(&"trial_system_value".to_string())
        );
        Ok(())
    }

    #[test]
    fn set_trial_state_values_complete() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize, Direction::Maximize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        assert_eq!(trial.state_values, TrialStateValues::Running);

        storage.set_trial_state_values(
            study_id,
            trial.number,
            TrialStateValues::Complete(vec![1.5, 2.5]),
        )?;

        let trial = storage.get_trial(study_id, trial.number)?;
        assert_eq!(
            trial.state_values,
            TrialStateValues::Complete(vec![1.5, 2.5])
        );
        Ok(())
    }

    #[test]
    fn set_trial_state_values_fail() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        storage.set_trial_state_values(study_id, trial.number, TrialStateValues::Fail)?;

        let trial = storage.get_trial(study_id, trial.number)?;
        assert_eq!(trial.state_values, TrialStateValues::Fail);
        Ok(())
    }

    #[test]
    fn get_trials_diff() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        // Create 5 trials
        for i in 0..5 {
            let trial = storage.create_new_trial(study_id)?;
            storage.set_trial_state_values(
                study_id,
                trial.number,
                TrialStateValues::Complete(vec![i as f64]),
            )?;
        }

        // Get all trials with number > 2
        let trials = storage.get_trials_diff(study_id, &[], 2)?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[0].number, 3);
        assert_eq!(trials[1].number, 4);

        // Get specific trials by number
        let trials = storage.get_trials_diff(study_id, &[0, 2], -1)?;
        assert_eq!(trials.len(), 5); // All trials + included ones

        // Get trials with number > 3 OR in [0, 1]
        let trials = storage.get_trials_diff(study_id, &[0, 1], 3)?;
        assert_eq!(trials.len(), 3); // trials 0, 1, 4
        Ok(())
    }

    #[test]
    fn run_optimization() -> Result<()> {
        let storage = SQLite3Storage::new(":memory:")?;
        storage.create_database()?;
        let storage = CachedStorage::new(Box::new(storage));

        let mut study = create_study("simple-quadratic", storage, vec![Direction::Minimize])?;
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
                println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
                Ok(vec![value])
            },
            sampler,
            100,
        )?;
        assert_eq!(study.get_trials()?.len(), 100);
        Ok(())
    }
}
