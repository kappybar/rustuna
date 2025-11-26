/// This crate specific `Error` type.
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
}
impl Error {
    pub fn new(kind: ErrorKind) -> Self {
        Error { kind }
    }
}

/// ErrorKind represents the kind of error.
#[derive(Debug, Clone)]
pub enum ErrorKind {
    ObjectiveError,
    SamplerError,
    StorageError,
    DuplicatedStudy,
    StudyNotFound,
    TrialNotFound,
    InvalidObjectiveValues,
    TrialAlreadyFinished,
    StorageInternalError,
    UnsupportedSearchSpace,
    UnsupportedMultiObjective,
    NoCompletedTrial,
    IncompatibleDistribution,
    Unexpected,
}
