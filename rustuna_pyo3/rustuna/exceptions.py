class RustunaError(Exception):
    """Base class for Rustuna specific errors."""

    pass


class DuplicatedStudyError(RustunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass
