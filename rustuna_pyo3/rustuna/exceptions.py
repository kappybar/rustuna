class RustunaError(Exception):
    """Base class for Rustuna specific errors.

    This is the base exception class for all Rustuna-related errors.
    All custom exceptions in Rustuna inherit from this class.
    """

    pass


class TrialPruned(RustunaError):
    """Exception for pruned trials."""

    pass


class StorageInternalError(RustunaError):
    """Exception for storage operation errors.

    This error is raised when an operation fails in the backend database of storage.
    Common causes include database connection errors, constraint violations,
    or other database-level failures.

    Example:
        This exception might be raised when:
        - Database connection is lost
        - SQL query execution fails
        - Database constraints are violated
    """

    pass


class DuplicatedStudyError(RustunaError):
    """Exception for duplicate study names.

    This error is raised when attempting to create a study with a name that
    already exists in the storage. Study names must be unique within a storage.

    Example:
        .. code-block:: python

            import rustuna

            storage = rustuna.Storage.in_memory()
            rustuna.create_study(study_name="my-study", storage=storage)
            # Raises DuplicatedStudyError
            rustuna.create_study(study_name="my-study", storage=storage)
    """

    pass


class UpdateFinishedTrialError(RustunaError, RuntimeError):
    """Exception for updating a finished trial.

    This error is raised when attempting to update a trial that has already
    finished (i.e., its state is COMPLETE, PRUNED, or FAIL). Finished trials
    should not be modified to maintain the integrity of the optimization history.

    Example:
        This exception might be raised when:
        - Calling :func:`~rustuna.Trial.suggest_float` after the trial has finished
        - Attempting to report values to a completed trial
    """

    pass
