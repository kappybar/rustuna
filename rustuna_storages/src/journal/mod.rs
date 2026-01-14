use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use rustuna_core::{Error, ErrorKind, Result};

pub mod file;
pub mod storage;

pub trait JournalBackend: Send + Sync {
    fn read_logs(
        &mut self,
        log_number_from: usize,
        handler: &mut dyn FnMut(JournalLog) -> Result<()>,
    ) -> Result<()>;
    fn append_logs(&mut self, logs: &[JournalLog]) -> Result<()>;
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl JournalOperation {
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
            _ => Err(Error::new(ErrorKind::StorageError)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalLog {
    pub op_code: i32,
    pub worker_id: String,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}
