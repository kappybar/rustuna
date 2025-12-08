class RustunaError(Exception):
    """Base class for Rustuna specific errors."""

    pass


class StorageInternalError(RustunaError):
    """Exception for storage operation.

    This error is raised when an operation failed in backend DB of storage.
    """

    pass


class DuplicatedStudyError(RustunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass


class UpdateFinishedTrialError(RustunaError, RuntimeError):
    """Exception for updating a finished trial.

    This error is raised when attempting to update a finished trial.
    """

    pass
