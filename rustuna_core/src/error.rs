use std::panic::Location;

/// This crate specific `Error` type.
#[derive(Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub reason: String,
    pub location: &'static Location<'static>,
}
impl Error {
    #[track_caller]
    pub fn new(kind: ErrorKind) -> Self {
        Self::with_reason(kind, String::new())
    }

    #[track_caller]
    pub fn with_reason<T: Into<String>>(kind: ErrorKind, reason: T) -> Self {
        Self {
            kind,
            reason: reason.into(),
            location: Location::caller(),
        }
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.reason.is_empty() {
            write!(f, "{:?}", self.kind)?;
        } else {
            write!(f, "{:?}: {}", self.kind, self.reason)?;
        }
        write!(f, " (at {}:{})", self.location.file(), self.location.line())?;
        Ok(())
    }
}

/// ErrorKind represents the kind of error.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind {
    ObjectiveError,
    SamplerError,
    StorageError,
    DuplicatedStudy,
    StudyNotFound,
    TrialNotFound,
    AttrOverwriteNotAllowed,
    InvalidObjectiveValues,
    TrialAlreadyFinished,
    StorageInternalError,
    UnsupportedSearchSpace,
    UnsupportedMultiObjective,
    NoCompletedTrial,
    IncompatibleDistribution,
    Unexpected,
}
