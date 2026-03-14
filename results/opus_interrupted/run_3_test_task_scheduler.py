"""Tests for the async task scheduler module."""

import asyncio
import time

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
    Task,
    TaskScheduler,
    TaskStatus,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_coro(result: object = "ok", delay: float = 0.0, fail_times: int = 0):
    """Return a coroutine factory that optionally fails *fail_times* before succeeding."""
    call_count = 0

    def factory():
        nonlocal call_count

        async def _work():
            nonlocal call_count
            call_count += 1
            if delay:
                await asyncio.sleep(delay)
            if call_count <= fail_times:
                raise RuntimeError(f"Intentional failure #{call_count}")
            return result

        return _work()

    return factory


# ------------------------------------------------------------------ #
# 1. Basic task execution
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_basic_task_execution():
    """Tasks without dependencies should execute and produce results."""
    scheduler = TaskScheduler(max_concurrency=2)

    scheduler.add_task(
        Task(id="a", name="Alpha", priority=5),
        _make_coro(result="alpha_result"),
    )
    scheduler.add_task(
        Task(id="b", name="Beta", priority=8),
        _make_coro(result="beta_result"),
    )

    metrics = await scheduler.run()

    assert scheduler.get_task("a").status == TaskStatus.COMPLETED
    assert scheduler.get_task("a").result == "alpha_result"
    assert scheduler.get_task("b").status == TaskStatus.COMPLETED
    assert scheduler.get_task("b").result == "beta_result"
    assert metrics.completed_count == 2
    assert metrics.failed_count == 0
    assert metrics.total_time > 0


# ------------------------------------------------------------------ #
# 2. Dependency resolution
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_dependency_resolution():
    """Tasks must not start before their dependencies have completed."""
    execution_order: list[str] = []

    def _tracked_coro(task_id: str, delay: float = 0.01):
        def factory():
            async def _work():
                await asyncio.sleep(delay)
                execution_order.append(task_id)
                return task_id

            return _work()

        return factory

    scheduler = TaskScheduler(max_concurrency=4)
    # c depends on a and b
    scheduler.add_task(Task(id="a", name="A"), _tracked_coro("a"))
    scheduler.add_task(Task(id="b", name="B"), _tracked_coro("b"))
    scheduler.add_task(
        Task(id="c", name="C", dependencies=["a", "b"]),
        _tracked_coro("c"),
    )

    await scheduler.run()

    # c must appear after both a and b
    assert execution_order.index("c") > execution_order.index("a")
    assert execution_order.index("c") > execution_order.index("b")
    assert scheduler.get_task("c").status == TaskStatus.COMPLETED

    # Execution plan should have two groups: [a, b] then [c]
    plan = scheduler.get_execution_plan()
    assert len(plan) == 2
    assert set(plan[0]) == {"a", "b"}
    assert plan[1] == ["c"]


# ------------------------------------------------------------------ #
# 3. Circular dependency detection
# ------------------------------------------------------------------ #


def test_circular_dependency_detection():
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = TaskScheduler()

    scheduler.add_task(
        Task(id="x", name="X", dependencies=["z"]),
        _make_coro(),
    )
    scheduler.add_task(
        Task(id="y", name="Y", dependencies=["x"]),
        _make_coro(),
    )
    scheduler.add_task(
        Task(id="z", name="Z", dependencies=["y"]),
        _make_coro(),
    )

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


@pytest.mark.asyncio
async def test_circular_dependency_blocks_run():
    """Calling run() on a cyclic graph must also raise."""
    scheduler = TaskScheduler()

    scheduler.add_task(
        Task(id="a", name="A", dependencies=["b"]),
        _make_coro(),
    )
    scheduler.add_task(
        Task(id="b", name="B", dependencies=["a"]),
        _make_coro(),
    )

    with pytest.raises(CircularDependencyError):
        await scheduler.run()


# ------------------------------------------------------------------ #
# 4. Retry logic with exponential backoff
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff():
    """A task that fails twice then succeeds should complete after 2 retries."""
    scheduler = TaskScheduler(max_concurrency=1, base_retry_delay=0.01)

    scheduler.add_task(
        Task(id="flaky", name="Flaky", max_retries=3),
        _make_coro(result="recovered", fail_times=2),
    )

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    task = scheduler.get_task("flaky")
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "recovered"
    assert task.retry_count == 2
    assert metrics.total_retries == 2

    # Exponential backoff: delay_1 = 0.01, delay_2 = 0.02 => total >= 0.03
    assert elapsed >= 0.02  # conservative lower bound


@pytest.mark.asyncio
async def test_task_fails_after_max_retries():
    """A task that always fails should be marked FAILED after max_retries."""
    events: list[str] = []
    scheduler = TaskScheduler(max_concurrency=1, base_retry_delay=0.005)
    scheduler.on_task_fail(lambda t: events.append(f"fail:{t.id}"))

    scheduler.add_task(
        Task(id="doomed", name="Doomed", max_retries=2),
        _make_coro(fail_times=999),  # always fails
    )

    metrics = await scheduler.run()

    task = scheduler.get_task("doomed")
    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 3  # initial + 2 retries = 3 attempts
    assert metrics.failed_count == 1
    assert "fail:doomed" in events


# ------------------------------------------------------------------ #
# 5. Concurrent execution respects concurrency limit
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_concurrency_limit():
    """No more than max_concurrency tasks should run at the same time."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    def _tracking_coro(task_id: str):
        def factory():
            async def _work():
                nonlocal max_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > max_concurrent:
                        max_concurrent = current_concurrent

                await asyncio.sleep(0.05)

                async with lock:
                    current_concurrent -= 1

                return task_id

            return _work()

        return factory

    concurrency_limit = 2
    scheduler = TaskScheduler(max_concurrency=concurrency_limit)

    for i in range(6):
        scheduler.add_task(
            Task(id=f"t{i}", name=f"Task {i}"),
            _tracking_coro(f"t{i}"),
        )

    await scheduler.run()

    assert max_concurrent <= concurrency_limit
    # We expect the semaphore was actually exercised (> 1 concurrent)
    assert max_concurrent == concurrency_limit


# ------------------------------------------------------------------ #
# 6. Observer pattern events
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_observer_events():
    """on_task_start, on_task_complete, and on_task_fail fire correctly."""
    events: list[str] = []

    scheduler = TaskScheduler(max_concurrency=2, base_retry_delay=0.005)
    scheduler.on_task_start(lambda t: events.append(f"start:{t.id}"))
    scheduler.on_task_complete(lambda t: events.append(f"complete:{t.id}"))
    scheduler.on_task_fail(lambda t: events.append(f"fail:{t.id}"))

    scheduler.add_task(Task(id="good", name="Good"), _make_coro(result="ok"))
    scheduler.add_task(
        Task(id="bad", name="Bad", max_retries=0),
        _make_coro(fail_times=999),
    )

    await scheduler.run()

    assert "start:good" in events
    assert "complete:good" in events
    assert "start:bad" in events
    assert "fail:bad" in events
    assert "complete:bad" not in events


# ------------------------------------------------------------------ #
# 7. Dependency failure cascades
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_dependency_failure_cascades():
    """If a dependency fails, downstream tasks should also fail."""
    scheduler = TaskScheduler(max_concurrency=2, base_retry_delay=0.005)

    scheduler.add_task(
        Task(id="base", name="Base", max_retries=0),
        _make_coro(fail_times=999),
    )
    scheduler.add_task(
        Task(id="child", name="Child", dependencies=["base"]),
        _make_coro(result="should_not_run"),
    )

    metrics = await scheduler.run()

    assert scheduler.get_task("base").status == TaskStatus.FAILED
    assert scheduler.get_task("child").status == TaskStatus.FAILED
    assert metrics.failed_count == 2


# ------------------------------------------------------------------ #
# 8. Edge cases
# ------------------------------------------------------------------ #


def test_duplicate_task_id_rejected():
    """Adding two tasks with the same ID should raise ValueError."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="dup", name="First"), _make_coro())
    with pytest.raises(ValueError, match="already exists"):
        scheduler.add_task(Task(id="dup", name="Second"), _make_coro())


def test_invalid_priority_rejected():
    """Priority outside 1-10 should raise ValueError."""
    with pytest.raises(ValueError, match="Priority"):
        Task(id="bad", name="Bad", priority=0)
    with pytest.raises(ValueError, match="Priority"):
        Task(id="bad", name="Bad", priority=11)
