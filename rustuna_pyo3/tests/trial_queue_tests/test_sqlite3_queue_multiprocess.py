"""
Test SQLite3TrialQueue with multiple processes.

This test verifies that SQLite3TrialQueue works correctly in a multiprocess
environment where multiple workers push and pop trials concurrently.
"""

import multiprocessing
import tempfile
import time
from pathlib import Path

import rustuna


def worker_push(
    db_path: str, study_id: int, trial_ids: list[int], worker_id: int
) -> None:
    """Worker process that pushes trial IDs to the queue."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    for trial_id in trial_ids:
        queue.push(trial_id)
        time.sleep(0.001)  # Small delay to simulate realistic workload


def worker_pop(
    db_path: str, study_id: int, num_trials: int, result_queue: multiprocessing.Queue
) -> None:
    """Worker process that pops trial IDs from the queue."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    popped_ids: list[int] = []

    while len(popped_ids) < num_trials:
        try:
            trial_id = queue.pop()
            popped_ids.append(trial_id)
        except Exception:
            # Queue might be temporarily empty or locked
            time.sleep(0.01)

    result_queue.put(popped_ids)


def test_sqlite3_queue_multiprocess_push_pop():
    """Test concurrent push and pop operations from multiple processes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "queue.db")
        study_id = 1

        # Create 3 workers, each pushing 10 trials
        num_workers = 3
        trials_per_worker = 10

        # Assign trial IDs to each worker
        all_trial_ids = list(range(1, num_workers * trials_per_worker + 1))
        worker_trial_ids = [
            all_trial_ids[i * trials_per_worker : (i + 1) * trials_per_worker]
            for i in range(num_workers)
        ]

        # Start push workers
        push_processes = [
            multiprocessing.Process(
                target=worker_push, args=(db_path, study_id, worker_trial_ids[i], i)
            )
            for i in range(num_workers)
        ]

        for p in push_processes:
            p.start()

        # Wait for all pushes to complete
        for p in push_processes:
            p.join()

        # Now pop all trials using multiple workers
        result_queue = multiprocessing.Queue()
        pop_processes = [
            multiprocessing.Process(
                target=worker_pop,
                args=(db_path, study_id, trials_per_worker, result_queue),
            )
            for _ in range(num_workers)
        ]

        for p in pop_processes:
            p.start()

        for p in pop_processes:
            p.join()

        # Collect all popped IDs
        all_popped = []
        while not result_queue.empty():
            all_popped.extend(result_queue.get())

        # Verify all trials were popped exactly once
        assert len(all_popped) == len(all_trial_ids)
        assert sorted(all_popped) == sorted(all_trial_ids)


def test_sqlite3_queue_fifo_across_processes():
    """Test that FIFO ordering is maintained across process boundaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "queue.db")
        study_id = 1

        # Push trials from one process
        trial_ids = [10, 20, 30, 40, 50]
        p = multiprocessing.Process(
            target=worker_push, args=(db_path, study_id, trial_ids, 0)
        )
        p.start()
        p.join()

        # Pop from another process
        queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
        popped_ids = []
        for _ in trial_ids:
            popped_ids.append(queue.pop())

        # Should maintain FIFO order
        assert popped_ids == trial_ids


def test_sqlite3_queue_no_duplicates():
    """Test that each trial is popped exactly once even with concurrent pop operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "queue.db")
        study_id = 1

        # Push 100 trials
        queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
        num_trials = 100
        for i in range(1, num_trials + 1):
            queue.push(i)

        # Pop with 5 concurrent workers
        num_workers = 5
        trials_per_worker = num_trials // num_workers

        result_queue = multiprocessing.Queue()
        pop_processes = [
            multiprocessing.Process(
                target=worker_pop,
                args=(db_path, study_id, trials_per_worker, result_queue),
            )
            for _ in range(num_workers)
        ]

        for p in pop_processes:
            p.start()

        for p in pop_processes:
            p.join()

        # Collect all popped IDs
        all_popped = []
        while not result_queue.empty():
            all_popped.extend(result_queue.get())

        # Verify no duplicates and all trials accounted for
        assert len(all_popped) == num_trials
        assert len(set(all_popped)) == num_trials  # No duplicates
        assert sorted(all_popped) == list(range(1, num_trials + 1))


def push_trials_for_persistence(
    db_path: str, study_id: int, result_queue: multiprocessing.Queue
) -> None:
    """Push trials for persistence test."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    for i in range(1, 11):
        queue.push(i)
    result_queue.put("done")


def pop_some_trials(
    db_path: str, study_id: int, count: int, result_queue: multiprocessing.Queue
) -> None:
    """Pop some trials for persistence test."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    result = [queue.pop() for _ in range(count)]
    result_queue.put(result)


def push_for_study(
    db_path: str,
    study_id: int,
    trial_ids: list[int],
    result_queue: multiprocessing.Queue,
) -> None:
    """Push trials for a specific study."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    for trial_id in trial_ids:
        queue.push(trial_id)
    result_queue.put("done")


def pop_all_from_study(
    db_path: str,
    study_id: int,
    expected_count: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Pop all trials from a specific study."""
    queue = rustuna.TrialQueue.sqlite3(db_path, study_id)
    popped = []
    for _ in range(expected_count):
        popped.append(queue.pop())
    result_queue.put((study_id, popped))


def test_sqlite3_queue_study_isolation():
    """Test that different studies have isolated queues even in multiprocess environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "queue.db")

        # Push trials for study 1 and 2 from different processes
        study1_ids = [1, 2, 3, 4, 5]
        study2_ids = [10, 20, 30, 40, 50]

        result_queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(
            target=push_for_study, args=(db_path, 1, study1_ids, result_queue)
        )
        p2 = multiprocessing.Process(
            target=push_for_study, args=(db_path, 2, study2_ids, result_queue)
        )

        p1.start()
        p2.start()
        p1.join()
        p2.join()
        result_queue.get()  # Wait for p1
        result_queue.get()  # Wait for p2

        # Pop from each study in different processes
        result_queue2 = multiprocessing.Queue()
        p1 = multiprocessing.Process(
            target=pop_all_from_study, args=(db_path, 1, len(study1_ids), result_queue2)
        )
        p2 = multiprocessing.Process(
            target=pop_all_from_study, args=(db_path, 2, len(study2_ids), result_queue2)
        )

        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # Collect results
        results = {}
        while not result_queue2.empty():
            study_id, popped = result_queue2.get()
            results[study_id] = popped

        # Verify isolation
        assert results[1] == study1_ids
        assert results[2] == study2_ids


def test_sqlite3_queue_persistence():
    """Test that queue contents persist across process restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "queue.db")
        study_id = 1

        # Push trials in one process
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=push_trials_for_persistence, args=(db_path, study_id, result_queue)
        )
        p.start()
        p.join()
        result_queue.get()  # Wait for completion

        # Pop some trials in another process
        result_queue1 = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=pop_some_trials, args=(db_path, study_id, 5, result_queue1)
        )
        p.start()
        p.join()
        first_batch = result_queue1.get()

        # Pop remaining trials in yet another process
        result_queue2 = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=pop_some_trials, args=(db_path, study_id, 5, result_queue2)
        )
        p.start()
        p.join()
        second_batch = result_queue2.get()

        # Verify all trials were popped in FIFO order
        all_popped = first_batch + second_batch
        assert all_popped == list(range(1, 11))


if __name__ == "__main__":
    test_sqlite3_queue_multiprocess_push_pop()
    test_sqlite3_queue_fifo_across_processes()
    test_sqlite3_queue_no_duplicates()
    test_sqlite3_queue_study_isolation()
    test_sqlite3_queue_persistence()
    print("All SQLite3TrialQueue multiprocess tests passed!")
