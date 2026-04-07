from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal, Self

import pytest

import rustuna

TrialQueueType = Literal["in_memory", "directory", "sqlite3"]
MultiprocessQueueType = Literal["directory", "sqlite3"]


class TrialQueueFactory:
    def __init__(self, queue_type: TrialQueueType) -> None:
        self.queue_type = queue_type
        self.tmpdir: str | None = None
        self.base_path: str | None = None
        self.study_id = 1
        self.queue: rustuna.TrialQueue | None = None

    def __enter__(self) -> Self:
        if self.queue_type == "in_memory":
            self.queue = rustuna.TrialQueue.in_memory()
        elif self.queue_type == "directory":
            self.tmpdir = tempfile.mkdtemp()
            self.base_path = str(Path(self.tmpdir) / "queue")
            self.queue = rustuna.TrialQueue.directory(self.base_path)
        elif self.queue_type == "sqlite3":
            self.tmpdir = tempfile.mkdtemp()
            self.base_path = str(Path(self.tmpdir) / "queue.db")
            self.queue = rustuna.TrialQueue.sqlite3(
                self.base_path, study_id=self.study_id
            )
        else:
            raise ValueError(f"Unknown queue type: {self.queue_type}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def create_queue(self, study_id: int | None = None) -> rustuna.TrialQueue:
        if self.queue_type == "in_memory":
            return rustuna.TrialQueue.in_memory()
        if self.base_path is None:
            raise RuntimeError(
                "TrialQueueFactory must be entered before creating queues"
            )
        resolved_study_id = self.study_id if study_id is None else study_id
        return make_trial_queue(self.queue_type, self.base_path, resolved_study_id)


def make_trial_queue(
    queue_type: MultiprocessQueueType | TrialQueueType,
    base_path: str,
    study_id: int = 1,
) -> rustuna.TrialQueue:
    if queue_type == "directory":
        return rustuna.TrialQueue.directory(base_path)
    if queue_type == "sqlite3":
        return rustuna.TrialQueue.sqlite3(base_path, study_id=study_id)
    if queue_type == "in_memory":
        return rustuna.TrialQueue.in_memory()
    raise ValueError(f"Unknown queue type: {queue_type}")


parametrize_trial_queue = pytest.mark.parametrize(
    "queue_type",
    ["in_memory", "directory", "sqlite3"],
    ids=["InMemoryTrialQueue", "DirectoryTrialQueue", "SQLite3TrialQueue"],
)

parametrize_multiprocess_trial_queue = pytest.mark.parametrize(
    "queue_type",
    ["directory", "sqlite3"],
    ids=["DirectoryTrialQueue", "SQLite3TrialQueue"],
)
