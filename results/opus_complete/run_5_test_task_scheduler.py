"""Tests for task_scheduler module."""

from __future__ import annotations

import asyncio
import time

import pytest

from task_scheduler import (
    CircularDependencyError,
    ExecutionMetrics,
    Task,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop(task: Task) -> str:
    """Trivial coroutine that returns immediately."""
    return f"{task.name}-done"


async def _slow(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


def _make_fail_after(n: int):
    """Return a coroutine factory that fails the first *n* calls then succeeds."""
    call_count = {"n": 0}

    async def _coro(task: Task) -> str:
        call_count["n"] += 1
        if call_count["n"] <= n:
            raise RuntimeError(f"transient failure #{call_count['n']}")
        return f"{task.name}-recovered"

    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution() -> None:
    """Tasks with no dependencies execute and reach COMPLETED status."""
    scheduler = TaskScheduler(max_concurrency=2)

    t1 = Task(id="a", name="Alpha", priority=5)
    t2 = Task(id="b", name="Beta", priority=8)

    scheduler.add_task(t1, _noop)
    scheduler.add_task(t2, _noop)

    metrics = await scheduler.run()

    assert t1.status == TaskStatus.COMPLETED
    assert t2.status == TaskStatus.COMPLETED
    assert t1.result == "Alpha-done"
    assert t2.result == "Beta-done"
    assert metrics.total_time > 0
    assert "a" in metrics.per_task_time
    assert "b" in metrics.per_task_time


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks execute only after their dependencies have completed."""
    execution_order: list[str] = []

    async def _track(task: Task) -> str:
        execution_order.append(task.id)
        return task.id

    scheduler = TaskScheduler(max_concurrency=1)

    # c depends on b, b depends on a  =>  a -> b -> c
    scheduler.add_task(Task(id="a", name="A"), _track)
    scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _track)
    scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _track)

    await scheduler.run()

    assert execution_order == ["a", "b", "c"]

    # Verify the execution plan groups
    scheduler2 = TaskScheduler()
    scheduler2.add_task(Task(id="a", name="A"), _noop)
    scheduler2.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
    scheduler2.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)

    plan = scheduler2.get_execution_plan()
    assert plan == [["a"], ["b"], ["c"]]


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circular_dependency_detection() -> None:
    """A cycle in the dependency graph raises CircularDependencyError."""
    scheduler = TaskScheduler()

    scheduler.add_task(Task(id="x", name="X", dependencies=["z"]), _noop)
    scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop)
    scheduler.add_task(Task(id="z", name="Z", dependencies=["y"]), _noop)

    with pytest.raises(CircularDependencyError):
        await scheduler.run()

    # Also verify via get_execution_plan
    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff() -> None:
    """A task that fails transiently is retried with increasing delays."""
    scheduler = TaskScheduler(max_concurrency=1, base_backoff=0.05)

    # Fails twice then succeeds on third attempt.
    coro = _make_fail_after(2)
    task = Task(id="r", name="Retry", max_retries=3)
    scheduler.add_task(task, coro)

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    assert task.status == TaskStatus.COMPLETED
    assert task.retry_count == 2
    assert task.result == "Retry-recovered"
    assert metrics.retry_counts["r"] == 2

    # Two retries: backoff = 0.05 + 0.10 = 0.15 s minimum
    assert elapsed >= 0.14  # small tolerance


@pytest.mark.asyncio
async def test_retry_exhaustion_marks_failed() -> None:
    """A task that exceeds max_retries ends up FAILED."""
    scheduler = TaskScheduler(max_concurrency=1, base_backoff=0.01)

    coro = _make_fail_after(100)  # always fails
    task = Task(id="f", name="Failing", max_retries=2)
    scheduler.add_task(task, coro)

    await scheduler.run()

    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 3  # initial + 2 retries = 3 attempts


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    """No more than max_concurrency tasks run simultaneously."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def _track_concurrency(task: Task) -> str:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrent -= 1
        return task.name

    scheduler = TaskScheduler(max_concurrency=2)

    for i in range(6):
        scheduler.add_task(
            Task(id=f"t{i}", name=f"Task{i}"),
            _track_concurrency,
        )

    await scheduler.run()

    assert max_concurrent <= 2, (
        f"Expected at most 2 concurrent tasks, observed {max_concurrent}"
    )
    # All tasks should have completed
    for i in range(6):
        assert scheduler._tasks[f"t{i}"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 6. Observer pattern events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events() -> None:
    """Event listeners are called on start, complete, and fail."""
    started: list[str] = []
    completed: list[str] = []
    failed: list[str] = []

    scheduler = TaskScheduler(max_concurrency=2, base_backoff=0.01)
    scheduler.on("on_task_start", lambda t: started.append(t.id))
    scheduler.on("on_task_complete", lambda t: completed.append(t.id))
    scheduler.on("on_task_fail", lambda t: failed.append(t.id))

    async def _fail(task: Task) -> None:
        raise RuntimeError("boom")

    scheduler.add_task(Task(id="ok", name="OK"), _noop)
    scheduler.add_task(Task(id="bad", name="BAD", max_retries=0), _fail)

    await scheduler.run()

    assert "ok" in started
    assert "ok" in completed
    assert "bad" in failed


# ---------------------------------------------------------------------------
# 7. Edge: unknown dependency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_dependency_raises() -> None:
    """Referencing a non-existent dependency raises ValueError."""
    scheduler = TaskScheduler()
    scheduler.add_task(
        Task(id="orphan", name="Orphan", dependencies=["ghost"]), _noop
    )

    with pytest.raises(ValueError, match="unknown task 'ghost'"):
        await scheduler.run()


# ---------------------------------------------------------------------------
# 8. Priority validation
# ---------------------------------------------------------------------------

def test_invalid_priority_raises() -> None:
    """Creating a Task with priority outside 1-10 raises ValueError."""
    with pytest.raises(ValueError):
        Task(id="p", name="P", priority=0)
    with pytest.raises(ValueError):
        Task(id="p", name="P", priority=11)
