"""Tests for the async task scheduler."""

from __future__ import annotations

import asyncio
import time

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coro(result: object = "ok", delay: float = 0.0, fail_times: int = 0):
    """Return an async callable that optionally fails *fail_times* then succeeds."""
    call_count = 0

    async def _coro():
        nonlocal call_count
        call_count += 1
        if delay:
            await asyncio.sleep(delay)
        if call_count <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count}")
        return result

    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution():
    """A single task with no dependencies should run and complete."""
    scheduler = TaskScheduler()
    task = Task(id="t1", name="Simple Task")
    scheduler.add_task(task, _make_coro(result=42))

    metrics = await scheduler.run()

    assert task.status == TaskStatus.COMPLETED
    assert task.result == 42
    assert "t1" in metrics.task_metrics
    assert metrics.task_metrics["t1"].duration >= 0
    assert metrics.total_time >= 0


# ---------------------------------------------------------------------------
# 2. Dependency resolution (topological ordering)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution():
    """Tasks must execute only after their dependencies complete.

    Graph:  t1 -> t2 -> t3
    Expected groups: [[t1], [t2], [t3]]
    """
    scheduler = TaskScheduler()
    execution_order: list[str] = []

    async def _tracking_coro(tid: str):
        execution_order.append(tid)
        return tid

    scheduler.add_task(Task(id="t1", name="Task 1"), lambda: _tracking_coro("t1"))
    scheduler.add_task(
        Task(id="t2", name="Task 2", dependencies=["t1"]),
        lambda: _tracking_coro("t2"),
    )
    scheduler.add_task(
        Task(id="t3", name="Task 3", dependencies=["t2"]),
        lambda: _tracking_coro("t3"),
    )

    plan = scheduler.get_execution_plan()
    assert plan == [["t1"], ["t2"], ["t3"]]

    await scheduler.run()

    assert execution_order == ["t1", "t2", "t3"]


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

def test_circular_dependency_detection():
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = TaskScheduler()
    scheduler.add_task(
        Task(id="a", name="A", dependencies=["c"]),
        _make_coro(),
    )
    scheduler.add_task(
        Task(id="b", name="B", dependencies=["a"]),
        _make_coro(),
    )
    scheduler.add_task(
        Task(id="c", name="C", dependencies=["b"]),
        _make_coro(),
    )

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


@pytest.mark.asyncio
async def test_circular_dependency_on_run():
    """run() should also raise if a cycle exists."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="x", name="X", dependencies=["y"]), _make_coro())
    scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), _make_coro())

    with pytest.raises(CircularDependencyError):
        await scheduler.run()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff():
    """A task that fails twice then succeeds should complete after retries.

    With max_retries=3 and fail_times=2 the task should ultimately succeed.
    Backoff delays: 0.5 s (retry 1) + 1.0 s (retry 2) = 1.5 s minimum.
    We use a generous lower bound to avoid flakiness.
    """
    scheduler = TaskScheduler()
    task = Task(id="flaky", name="Flaky Task", max_retries=3)
    scheduler.add_task(task, _make_coro(result="recovered", fail_times=2))

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    assert task.status == TaskStatus.COMPLETED
    assert task.result == "recovered"
    assert task.retry_count == 2
    assert metrics.task_metrics["flaky"].retries == 2
    # At least 1.0 s of backoff should have elapsed (0.5 + 1.0 = 1.5, allow margin).
    assert elapsed >= 1.0


@pytest.mark.asyncio
async def test_retry_exhaustion():
    """A task that always fails should end up FAILED after max_retries."""
    scheduler = TaskScheduler()
    task = Task(id="bad", name="Always Fails", max_retries=2)
    scheduler.add_task(task, _make_coro(fail_times=999))

    fail_events: list[Task] = []
    scheduler.on("on_task_fail", lambda t, exc: fail_events.append(t))

    await scheduler.run()

    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 2
    assert len(fail_events) == 1


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit():
    """No more than *max_concurrency* tasks should run at the same time.

    We create 6 independent tasks each taking 0.2 s and set max_concurrency=2.
    Total wall-clock time should be >= 0.6 s (3 batches of 2).
    If concurrency were unbounded it would be ~0.2 s.
    """
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def _tracked():
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        await asyncio.sleep(0.2)
        async with lock:
            current_concurrent -= 1

    scheduler = TaskScheduler(max_concurrency=2)
    for i in range(6):
        scheduler.add_task(Task(id=f"c{i}", name=f"C{i}"), _tracked)

    metrics = await scheduler.run()

    assert max_concurrent <= 2
    # 6 tasks / 2 concurrency * 0.2 s = ~0.6 s
    assert metrics.total_time >= 0.5


# ---------------------------------------------------------------------------
# 6. Observer events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events():
    """on_task_start and on_task_complete events should fire for each task."""
    scheduler = TaskScheduler()

    started: list[str] = []
    completed: list[str] = []

    scheduler.on("on_task_start", lambda t: started.append(t.id))
    scheduler.on("on_task_complete", lambda t: completed.append(t.id))

    scheduler.add_task(Task(id="e1", name="E1"), _make_coro())
    scheduler.add_task(Task(id="e2", name="E2"), _make_coro())

    await scheduler.run()

    assert set(started) == {"e1", "e2"}
    assert set(completed) == {"e1", "e2"}


# ---------------------------------------------------------------------------
# 7. Failed dependency propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_dependency_skips_dependents():
    """If a dependency fails, its dependents should also be marked FAILED."""
    scheduler = TaskScheduler()

    scheduler.add_task(
        Task(id="root", name="Root", max_retries=1),
        _make_coro(fail_times=999),
    )
    scheduler.add_task(
        Task(id="child", name="Child", dependencies=["root"]),
        _make_coro(),
    )

    await scheduler.run()

    assert scheduler._tasks["root"].status == TaskStatus.FAILED
    assert scheduler._tasks["child"].status == TaskStatus.FAILED
