"""
Test DirectoryTrialQueue with multiple processes.

This test verifies that DirectoryTrialQueue works correctly in a multiprocess
environment where multiple workers push and pop trials concurrently.
"""

import multiprocessing
import tempfile
import time
from pathlib import Path

import rustuna


def worker_push(queue_dir: str, trial_ids: list[int], worker_id: int) -> None:
    """Worker process that pushes trial IDs to the queue."""
    queue = rustuna.TrialQueue.directory(queue_dir)
    for trial_id in trial_ids:
        queue.push(trial_id)
        time.sleep(0.001)  # Small delay to simulate realistic workload


def worker_pop(
    queue_dir: str, num_trials: int, result_queue: multiprocessing.Queue
) -> None:
    """Worker process that pops trial IDs from the queue."""
    queue = rustuna.TrialQueue.directory(queue_dir)
    popped_ids: list[int] = []

    while len(popped_ids) < num_trials:
        try:
            trial_id = queue.pop()
            popped_ids.append(trial_id)
        except Exception:
            # Queue might be temporarily empty
            time.sleep(0.01)

    result_queue.put(popped_ids)


def test_directory_queue_multiprocess_push_pop():
    """Test concurrent push and pop operations from multiple processes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = str(Path(tmpdir) / "queue")

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
                target=worker_push, args=(queue_dir, worker_trial_ids[i], i)
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
                target=worker_pop, args=(queue_dir, trials_per_worker, result_queue)
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


def test_directory_queue_fifo_across_processes():
    """Test that FIFO ordering is maintained across process boundaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = str(Path(tmpdir) / "queue")

        # Push trials from one process
        trial_ids = [10, 20, 30, 40, 50]
        p = multiprocessing.Process(target=worker_push, args=(queue_dir, trial_ids, 0))
        p.start()
        p.join()

        # Pop from another process
        queue = rustuna.TrialQueue.directory(queue_dir)
        popped_ids = []
        for _ in trial_ids:
            popped_ids.append(queue.pop())

        # Should maintain FIFO order
        assert popped_ids == trial_ids


def test_directory_queue_no_duplicates():
    """Test that each trial is popped exactly once even with concurrent pop operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = str(Path(tmpdir) / "queue")

        # Push 100 trials
        queue = rustuna.TrialQueue.directory(queue_dir)
        num_trials = 100
        for i in range(1, num_trials + 1):
            queue.push(i)

        # Pop with 5 concurrent workers
        num_workers = 5
        trials_per_worker = num_trials // num_workers

        result_queue = multiprocessing.Queue()
        pop_processes = [
            multiprocessing.Process(
                target=worker_pop, args=(queue_dir, trials_per_worker, result_queue)
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
    queue_dir: str, result_queue: multiprocessing.Queue
) -> None:
    """Push trials for persistence test."""
    queue = rustuna.TrialQueue.directory(queue_dir)
    for i in range(1, 11):
        queue.push(i)
    result_queue.put("done")


def pop_some_trials(
    queue_dir: str, count: int, result_queue: multiprocessing.Queue
) -> None:
    """Pop some trials for persistence test."""
    queue = rustuna.TrialQueue.directory(queue_dir)
    result = [queue.pop() for _ in range(count)]
    result_queue.put(result)


def test_directory_queue_persistence():
    """Test that queue contents persist across process restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = str(Path(tmpdir) / "queue")

        # Push trials in one process
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=push_trials_for_persistence, args=(queue_dir, result_queue)
        )
        p.start()
        p.join()
        result_queue.get()  # Wait for completion

        # Pop some trials in another process
        result_queue1 = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=pop_some_trials, args=(queue_dir, 5, result_queue1)
        )
        p.start()
        p.join()
        first_batch = result_queue1.get()

        # Pop remaining trials in yet another process
        result_queue2 = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=pop_some_trials, args=(queue_dir, 5, result_queue2)
        )
        p.start()
        p.join()
        second_batch = result_queue2.get()

        # Verify all trials were popped in FIFO order
        all_popped = first_batch + second_batch
        assert all_popped == list(range(1, 11))


if __name__ == "__main__":
    test_directory_queue_multiprocess_push_pop()
    test_directory_queue_fifo_across_processes()
    test_directory_queue_no_duplicates()
    test_directory_queue_persistence()
    print("All DirectoryTrialQueue multiprocess tests passed!")
