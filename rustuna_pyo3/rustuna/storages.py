from __future__ import annotations

from rustuna._protocols import StorageProtocol
from rustuna._rustuna import Storage, JournalFileStorage

__all__ = [
    "InMemoryStorage",
    "JournalFileStorage",
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
