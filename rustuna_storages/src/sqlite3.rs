use std::collections::HashMap;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use rustuna_core::attr::{AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage_cache::CachedStorageBackend;
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
            let (distribution, _labels) = parse_distribution_json(&distribution_json)?;
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

fn parse_distribution_json(
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
    use rustuna_core::distribution::Distribution;
    use rustuna_core::study::Direction;

    fn init_storage() -> SQLite3Storage {
        let storage = SQLite3Storage::new(":memory:").unwrap();
        storage.create_database().unwrap();
        storage
    }
}
