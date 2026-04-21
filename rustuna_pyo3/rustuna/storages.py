from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rustuna._rustuna import OptunaStorageProtocol, StorageProtocol

from rustuna._rustuna import Storage

__all__ = [
    "InMemoryStorage",
    "JournalStorage",
    "OptunaStorageProtocol",
    "SQLite3Storage",
    "StorageProtocol",
]


# TODO(c-bata): Replace InMemoryStorage with a Python concrete class.
def InMemoryStorage() -> StorageProtocol:
    return Storage.in_memory()


# TODO(c-bata): Replace JournalFileStorage with a Python concrete class.
def JournalFileStorage(file_path: str) -> OptunaStorageProtocol:
    return Storage.journal_file(file_path)


# TODO(c-bata): Replace InMemoryStorage with a Python concrete class.
def SQLite3Storage(
    file_path: str, *, create_database: bool = True
) -> OptunaStorageProtocol:
    return Storage.sqlite3(file_path=file_path, create_database=create_database)
