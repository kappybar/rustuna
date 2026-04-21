from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rustuna._rustuna import TrialQueueProtocol

from rustuna._rustuna import TrialQueue

__all__ = [
    "InMemoryTrialQueue",
    "SQLite3TrialQueue",
    "DirectoryTrialQueue",
]


# TODO(c-bata): Replace InMemoryTrialQueue with a Python concrete class.
def InMemoryTrialQueue() -> TrialQueueProtocol:
    """An in-memory TrialQueue implementation.

    This queue stores trial IDs in memory and does not persist across process restarts.
    Suitable for single-process optimization or when persistence is not required.

    Returns:
        An in-memory trial queue instance.
    """
    return TrialQueue.in_memory()


# TODO(c-bata): Replace SQLite3TrialQueue with a Python concrete class.
def SQLite3TrialQueue(db_path: str, study_id: int) -> TrialQueueProtocol:
    """An SQLite3 based TrialQueue implementation.

    This queue uses SQLite to persist trial IDs with ACID guarantees. Multiple studies
    can share the same database file, with study_id used for isolation.

    Args:
        db_path: Path to the SQLite database file.
        study_id: Study ID to isolate trials for this queue.

    Returns:
        An SQLite3-based trial queue instance.
    """
    return TrialQueue.sqlite3(db_path=db_path, study_id=study_id)


# TODO(c-bata): Replace SQLite3TrialQueue with a Python concrete class.
def DirectoryTrialQueue(base_dir: str) -> TrialQueueProtocol:
    """A directory-based trial queue.

    This queue uses the filesystem to persist trial IDs and provides multi-process
    safety through atomic file operations. The queue is stored in two subdirectories
    under the base directory: 'pending/' for queued trials and 'processing/' for
    trials being processed.

    Args:
        base_dir: Base directory path for the queue. Should be study-specific
            (e.g., '{storage_dir}/queue/{study_id}/') to ensure isolation between studies.

    Returns:
        A directory-based trial queue instance.
    """
    return TrialQueue.directory(base_dir=base_dir)
