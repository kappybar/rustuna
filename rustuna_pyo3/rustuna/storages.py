from __future__ import annotations

from rustuna._protocols import StorageProtocol
from rustuna._rustuna import Storage

__all__ = [
    "InMemoryStorage",
    "JournalStorage",
    "SQLite3Storage",
    "StorageProtocol",
]


# TODO(c-bata): Replace InMemoryStorage with a Python concrete class.
def InMemoryStorage() -> StorageProtocol:
    """Create an in-memory storage.

    Returns:
        An in-memory storage instance.
    """
    return Storage.in_memory()


# TODO(c-bata): Replace JournalFileStorage with a Python concrete class.
def JournalFileStorage(
    file_path: str,
    *,
    load_discarded_trials: bool = False,
) -> StorageProtocol:
    """Create a Journal storage with its file backend.

    Args:
        file_path: Path to the journal log file.
        load_discarded_trials: If True, ignore discard logs and load all trials.

    Returns:
        A Journal storage instance.
    """
    return Storage.journal_file(
        file_path,
        load_discarded_trials=load_discarded_trials,
    )


# TODO(c-bata): Replace InMemoryStorage with a Python concrete class.
def SQLite3Storage(file_path: str, *, create_database: bool = True) -> StorageProtocol:
    """Create a SQLite3 storage.

    Args:
        file_path: Path to the SQLite3 database file.
        create_database: If True, initialize the database when it is missing.

    Returns:
        A SQLite3 storage instance.
    """
    return Storage.sqlite3(file_path=file_path, create_database=create_database)
