from __future__ import annotations

import time
from concurrent.futures import Future, ProcessPoolExecutor

import pytest

from . import (
    MultiprocessQueueType,
    TrialQueueFactory,
    make_trial_queue,
    parametrize_multiprocess_trial_queue,
)


def _make_queue(
    queue_type: MultiprocessQueueType,
    base_path: str,
    study_id: int,
):
    return make_trial_queue(queue_type, base_path, study_id)


def _worker_push(
    queue_type: MultiprocessQueueType,
    base_path: str,
    study_id: int,
    trial_ids: list[int],
) -> None:
    queue = _make_queue(queue_type, base_path, study_id)
    for trial_id in trial_ids:
        queue.push(trial_id)
        time.sleep(0.001)


def _worker_pop_exact(
    queue_type: MultiprocessQueueType,
    base_path: str,
    study_id: int,
    count: int,
) -> list[int]:
    queue = _make_queue(queue_type, base_path, study_id)
    popped_ids: list[int] = []

    while len(popped_ids) < count:
        try:
            popped_ids.append(queue.pop())
        except Exception:
            time.sleep(0.01)
    return popped_ids


def _worker_pop_all_for_study(
    queue_type: MultiprocessQueueType,
    base_path: str,
    study_id: int,
    expected_count: int,
):
    return study_id, _worker_pop_exact(queue_type, base_path, study_id, expected_count)


@parametrize_multiprocess_trial_queue
def test_queue_multiprocess_push_pop(
    queue_type: MultiprocessQueueType,
) -> None:
    with TrialQueueFactory(queue_type) as factory:
        assert factory.base_path is not None
        num_workers = 3
        trials_per_worker = 10
        all_trial_ids = list(range(1, num_workers * trials_per_worker + 1))
        worker_trial_ids = [
            all_trial_ids[i * trials_per_worker : (i + 1) * trials_per_worker]
            for i in range(num_workers)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            push_futures: list[Future[None]] = [
                executor.submit(
                    _worker_push,
                    queue_type,
                    factory.base_path,
                    factory.study_id,
                    worker_trial_ids[i],
                )
                for i in range(num_workers)
            ]
            for future in push_futures:
                future.result()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pop_futures: list[Future[list[int]]] = [
                executor.submit(
                    _worker_pop_exact,
                    queue_type,
                    factory.base_path,
                    factory.study_id,
                    trials_per_worker,
                )
                for _ in range(num_workers)
            ]
            all_popped = [
                trial_id for future in pop_futures for trial_id in future.result()
            ]

    assert len(all_popped) == len(all_trial_ids)
    assert sorted(all_popped) == sorted(all_trial_ids)


@parametrize_multiprocess_trial_queue
def test_queue_lifo_across_processes(
    queue_type: MultiprocessQueueType,
) -> None:
    with TrialQueueFactory(queue_type) as factory:
        assert factory.base_path is not None
        trial_ids = [10, 20, 30, 40, 50]

        with ProcessPoolExecutor(max_workers=1) as executor:
            executor.submit(
                _worker_push,
                queue_type,
                factory.base_path,
                factory.study_id,
                trial_ids,
            ).result()

        queue = factory.create_queue()
        popped_ids = [queue.pop() for _ in trial_ids]
    assert popped_ids == list(reversed(trial_ids))


@parametrize_multiprocess_trial_queue
def test_queue_no_duplicates(
    queue_type: MultiprocessQueueType,
) -> None:
    with TrialQueueFactory(queue_type) as factory:
        assert factory.queue is not None
        assert factory.base_path is not None
        num_trials = 100
        for i in range(1, num_trials + 1):
            factory.queue.push(i)

        num_workers = 5
        trials_per_worker = num_trials // num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pop_futures: list[Future[list[int]]] = [
                executor.submit(
                    _worker_pop_exact,
                    queue_type,
                    factory.base_path,
                    factory.study_id,
                    trials_per_worker,
                )
                for _ in range(num_workers)
            ]
            all_popped = [
                trial_id for future in pop_futures for trial_id in future.result()
            ]

    assert len(all_popped) == num_trials
    assert len(set(all_popped)) == num_trials
    assert sorted(all_popped) == list(range(1, num_trials + 1))


@parametrize_multiprocess_trial_queue
def test_queue_persistence(
    queue_type: MultiprocessQueueType,
) -> None:
    with TrialQueueFactory(queue_type) as factory:
        assert factory.base_path is not None
        trial_ids = list(range(1, 11))

        with ProcessPoolExecutor(max_workers=1) as executor:
            executor.submit(
                _worker_push,
                queue_type,
                factory.base_path,
                factory.study_id,
                trial_ids,
            ).result()
            first_batch = executor.submit(
                _worker_pop_exact,
                queue_type,
                factory.base_path,
                factory.study_id,
                5,
            ).result()
            second_batch = executor.submit(
                _worker_pop_exact,
                queue_type,
                factory.base_path,
                factory.study_id,
                5,
            ).result()

    assert first_batch + second_batch == list(range(10, 0, -1))


@parametrize_multiprocess_trial_queue
def test_queue_study_isolation(
    queue_type: MultiprocessQueueType,
) -> None:
    if queue_type != "sqlite3":
        pytest.skip("Study isolation is specific to SQLite3TrialQueue")

    with TrialQueueFactory(queue_type) as factory:
        assert factory.base_path is not None
        study1_ids = [1, 2, 3, 4, 5]
        study2_ids = [10, 20, 30, 40, 50]

        with ProcessPoolExecutor(max_workers=2) as executor:
            push_futures = [
                executor.submit(
                    _worker_push,
                    queue_type,
                    factory.base_path,
                    1,
                    study1_ids,
                ),
                executor.submit(
                    _worker_push,
                    queue_type,
                    factory.base_path,
                    2,
                    study2_ids,
                ),
            ]
            for future in push_futures:
                future.result()

        with ProcessPoolExecutor(max_workers=2) as executor:
            pop_futures = [
                executor.submit(
                    _worker_pop_all_for_study,
                    queue_type,
                    factory.base_path,
                    1,
                    len(study1_ids),
                ),
                executor.submit(
                    _worker_pop_all_for_study,
                    queue_type,
                    factory.base_path,
                    2,
                    len(study2_ids),
                ),
            ]
            results = dict(future.result() for future in pop_futures)

        assert results[1] == list(reversed(study1_ids))
        assert results[2] == list(reversed(study2_ids))
