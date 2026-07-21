from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import pytest

import rustuna

if TYPE_CHECKING:
    from rustuna.trial_queue import TrialQueueProtocol

TrialQueueType = Literal["in_memory", "python_in_memory", "directory", "sqlite3"]
MultiprocessQueueType = Literal["directory", "sqlite3"]


class DummyInMemoryTrialQueue:
    def __init__(self) -> None:
        self.queue: list[int] = []

    def enqueue(self, trial_id: int) -> None:
        self.queue.append(trial_id)

    def dequeue(self) -> int:
        if not self.queue:
            raise RuntimeError("queue is empty")
        return self.queue.pop(0)


class TrialQueueFactory:
    def __init__(self, queue_type: TrialQueueType) -> None:
        self.queue_type = queue_type
        self.tmpdir: str | None = None
        self.base_path: str | None = None
        self.namespace = "study-1"
        self.queue: TrialQueueProtocol | None = None

    def __enter__(self) -> Self:
        if self.queue_type == "in_memory":
            self.queue = rustuna.trial_queue.InMemoryTrialQueue()
        elif self.queue_type == "python_in_memory":
            self.queue = DummyInMemoryTrialQueue()
        elif self.queue_type == "directory":
            self.tmpdir = tempfile.mkdtemp()
            self.base_path = str(Path(self.tmpdir) / "queue")
            self.queue = rustuna.trial_queue.DirectoryTrialQueue(self.base_path)
        elif self.queue_type == "sqlite3":
            self.tmpdir = tempfile.mkdtemp()
            self.base_path = str(Path(self.tmpdir) / "queue.db")
            self.queue = rustuna.trial_queue.SQLite3TrialQueue(
                self.base_path, namespace=self.namespace
            )
        else:
            raise ValueError(f"Unknown queue type: {self.queue_type}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def create_queue(self, namespace: str | None = None) -> TrialQueueProtocol:
        if self.queue_type == "in_memory":
            return rustuna.trial_queue.InMemoryTrialQueue()
        if self.queue_type == "python_in_memory":
            return DummyInMemoryTrialQueue()
        if self.base_path is None:
            raise RuntimeError(
                "TrialQueueFactory must be entered before creating queues"
            )
        resolved_namespace = self.namespace if namespace is None else namespace
        return make_trial_queue(self.queue_type, self.base_path, resolved_namespace)


def make_trial_queue(
    queue_type: MultiprocessQueueType | TrialQueueType,
    base_path: str,
    namespace: str = "study-1",
) -> TrialQueueProtocol:
    if queue_type == "directory":
        return rustuna.trial_queue.DirectoryTrialQueue(base_path)
    if queue_type == "sqlite3":
        return rustuna.trial_queue.SQLite3TrialQueue(base_path, namespace=namespace)
    if queue_type == "in_memory":
        return rustuna.trial_queue.InMemoryTrialQueue()
    if queue_type == "python_in_memory":
        return DummyInMemoryTrialQueue()
    raise ValueError(f"Unknown queue type: {queue_type}")


parametrize_trial_queue = pytest.mark.parametrize(
    "queue_type",
    ["in_memory", "python_in_memory", "directory", "sqlite3"],
    ids=[
        "InMemoryTrialQueue",
        "DummyInMemoryTrialQueue",
        "DirectoryTrialQueue",
        "SQLite3TrialQueue",
    ],
)

parametrize_multiprocess_trial_queue = pytest.mark.parametrize(
    "queue_type",
    ["directory", "sqlite3"],
    ids=["DirectoryTrialQueue", "SQLite3TrialQueue"],
)
