use std::collections::HashMap;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;

use rustuna_core::{Error, ErrorKind, Result};

pub mod file;
pub mod storage;

/// Backend interface for journal-based storage.
///
/// Journal storage persists a log of storage operations and reconstructs the latest state by
/// replaying those logs. Backends implementing this trait only need to support appending logs and
/// reading them back in order.
pub trait JournalBackend: Send + Sync {
    /// Reads logs starting from `log_number_from` and passes them to `handler` in order.
    fn read_logs(
        &mut self,
        log_number_from: usize,
        handler: &mut dyn FnMut(JournalLog) -> Result<()>,
    ) -> Result<()>;
    /// Appends one or more logs atomically if possible.
    fn append_logs(&mut self, logs: &[JournalLog]) -> Result<()>;
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Operation kinds recorded in the journal log.
pub enum JournalOperation {
    CreateStudy = 0,
    DeleteStudy = 1,
    SetStudyUserAttr = 2,
    SetStudySystemAttr = 3,
    CreateTrial = 4,
    SetTrialParam = 5,
    SetTrialStateValues = 6,
    SetTrialIntermediateValue = 7,
    SetTrialUserAttr = 8,
    SetTrialSystemAttr = 9,
    DiscardTrials = 10,
}

impl JournalOperation {
    /// Decodes an operation code stored in a journal log.
    pub fn from_i32(value: i32) -> Result<Self> {
        match value {
            0 => Ok(JournalOperation::CreateStudy),
            1 => Ok(JournalOperation::DeleteStudy),
            2 => Ok(JournalOperation::SetStudyUserAttr),
            3 => Ok(JournalOperation::SetStudySystemAttr),
            4 => Ok(JournalOperation::CreateTrial),
            5 => Ok(JournalOperation::SetTrialParam),
            6 => Ok(JournalOperation::SetTrialStateValues),
            7 => Ok(JournalOperation::SetTrialIntermediateValue),
            8 => Ok(JournalOperation::SetTrialUserAttr),
            9 => Ok(JournalOperation::SetTrialSystemAttr),
            10 => Ok(JournalOperation::DiscardTrials),
            _ => Err(Error::new(ErrorKind::StorageError)),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
/// One operation record written to a journal backend.
pub struct JournalLog {
    pub op_code: i32,
    pub worker_id: String,
    // Design note: keep raw JSON to avoid eager parsing of user_attrs (dict[str, str] in Rustuna)
    // and preserve Optuna journal schema while letting Python-side conversion handle JSON types.
    #[serde(flatten)]
    pub fields: HashMap<String, Box<RawValue>>,
}

impl JournalLog {
    /// Returns the worker identifier that produced this log record.
    pub fn worker_id(&self) -> &str {
        self.worker_id.as_str()
    }
}

struct JournalLogVisitor;

impl<'de> Visitor<'de> for JournalLogVisitor {
    type Value = JournalLog;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a journal log object")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut op_code: Option<i32> = None;
        let mut worker_id: Option<String> = None;
        let mut fields: HashMap<String, Box<RawValue>> = HashMap::new();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "op_code" => op_code = Some(map.next_value()?),
                "worker_id" => worker_id = Some(map.next_value()?),
                _ => {
                    let value = map.next_value::<Box<RawValue>>()?;
                    fields.insert(key, value);
                }
            }
        }

        let op_code =
            op_code.ok_or_else(|| serde::de::Error::custom("Missing op_code in journal log"))?;
        let worker_id = worker_id
            .ok_or_else(|| serde::de::Error::custom("Missing worker_id in journal log"))?;

        Ok(JournalLog {
            op_code,
            worker_id,
            fields,
        })
    }
}

impl<'de> Deserialize<'de> for JournalLog {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(JournalLogVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::JournalLog;

    #[test]
    fn journal_log_typed_preserves_user_attr_string() {
        let raw = r#"{"op_code":8,"worker_id":"w","trial_id":1,"user_attr":{"k":"{\"a\":1}"}}"#;
        let log: JournalLog = serde_json::from_str(raw).expect("deserialize log");
        let user_attr = log.fields.get("user_attr").expect("user_attr");
        assert_eq!(user_attr.get(), "{\"k\":\"{\\\"a\\\":1}\"}");
    }
}
