"""
Unit tests for task_scheduler module.

Run with:  pytest test_task_scheduler.py -v
"""

from __future__ import annotations

import asyncio
import time

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop(task: Task) -> str:
    """A trivial coroutine that always succeeds."""
    return f"{task.name}-done"


async def _slow(task: Task) -> str:
    """A coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


_fail_counter: dict[str, int] = {}


def _make_flaky(task_id: str, fail_times: int):
    """Return a coroutine factory that fails *fail_times* before succeeding."""

    async def _coro(task: Task) -> str:
        _fail_counter.setdefault(task_id, 0)
        _fail_counter[task_id] += 1
        if _fail_counter[task_id] <= fail_times:
            raise RuntimeError(f"transient failure #{_fail_counter[task_id]}")
        return "recovered"

    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution() -> None:
    """Tasks with no dependencies execute and produce results."""
    scheduler = TaskScheduler()

    t1 = Task(id="t1", name="TaskOne", priority=5)
    t2 = Task(id="t2", name="TaskTwo", priority=8)

    scheduler.add_task(t1, _noop)
    scheduler.add_task(t2, _noop)

    metrics = await scheduler.run()

    assert t1.status == TaskStatus.COMPLETED
    assert t2.status == TaskStatus.COMPLETED
    assert t1.result == "TaskOne-done"
    assert t2.result == "TaskTwo-done"
    assert metrics.total_time > 0
    assert "t1" in metrics.task_metrics
    assert "t2" in metrics.task_metrics


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks execute only after their dependencies have completed."""
    execution_order: list[str] = []

    async def _track(task: Task) -> str:
        execution_order.append(task.id)
        return "ok"

    scheduler = TaskScheduler(max_concurrency=1)  # serial execution

    # t2 depends on t1; t3 depends on t2
    t1 = Task(id="t1", name="First", priority=5)
    t2 = Task(id="t2", name="Second", priority=5, dependencies=["t1"])
    t3 = Task(id="t3", name="Third", priority=5, dependencies=["t2"])

    scheduler.add_task(t1, _track)
    scheduler.add_task(t2, _track)
    scheduler.add_task(t3, _track)

    await scheduler.run()

    assert execution_order == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_execution_plan_groups() -> None:
    """get_execution_plan returns correct topological groups."""
    scheduler = TaskScheduler()

    t1 = Task(id="t1", name="A", priority=5)
    t2 = Task(id="t2", name="B", priority=5)
    t3 = Task(id="t3", name="C", priority=5, dependencies=["t1", "t2"])

    scheduler.add_task(t1, _noop)
    scheduler.add_task(t2, _noop)
    scheduler.add_task(t3, _noop)

    plan = scheduler.get_execution_plan()
    # t1 and t2 should be in the first group, t3 in the second
    assert len(plan) == 2
    assert set(plan[0]) == {"t1", "t2"}
    assert plan[1] == ["t3"]


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circular_dependency_detection() -> None:
    """CircularDependencyError is raised when cycles exist."""
    scheduler = TaskScheduler()

    t1 = Task(id="t1", name="A", dependencies=["t2"])
    t2 = Task(id="t2", name="B", dependencies=["t3"])
    t3 = Task(id="t3", name="C", dependencies=["t1"])

    scheduler.add_task(t1, _noop)
    scheduler.add_task(t2, _noop)
    scheduler.add_task(t3, _noop)

    with pytest.raises(CircularDependencyError):
        await scheduler.run()


def test_circular_dependency_in_plan() -> None:
    """get_execution_plan also detects cycles."""
    scheduler = TaskScheduler()
    t1 = Task(id="a", name="A", dependencies=["b"])
    t2 = Task(id="b", name="B", dependencies=["a"])
    scheduler.add_task(t1, _noop)
    scheduler.add_task(t2, _noop)

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff() -> None:
    """A task that fails transiently is retried with increasing delays."""
    _fail_counter.clear()

    scheduler = TaskScheduler(base_backoff=0.01)
    task = Task(id="flaky", name="Flaky", max_retries=3)
    scheduler.add_task(task, _make_flaky("flaky", fail_times=2))

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    assert task.status == TaskStatus.COMPLETED
    assert task.result == "recovered"
    assert task.retry_count == 2
    # Backoff: 0.01 + 0.02 = 0.03 s minimum extra delay
    assert elapsed >= 0.03


@pytest.mark.asyncio
async def test_retry_exhaustion_marks_failed() -> None:
    """A task exceeding max_retries ends up FAILED."""
    _fail_counter.clear()

    scheduler = TaskScheduler(base_backoff=0.005)
    task = Task(id="doomed", name="Doomed", max_retries=2)
    scheduler.add_task(task, _make_flaky("doomed", fail_times=99))

    events: list[str] = []
    scheduler.on("on_task_fail", lambda t, exc: events.append(f"fail:{t.id}"))

    await scheduler.run()

    assert task.status == TaskStatus.FAILED
    assert "fail:doomed" in events


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit_respected() -> None:
    """No more than max_concurrency tasks run simultaneously."""
    max_conc = 2
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def _measure(task: Task) -> str:
        nonlocal current, peak
        async with lock:
            current += 1
            if current > peak:
                peak = current
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return "done"

    scheduler = TaskScheduler(max_concurrency=max_conc)
    for i in range(6):
        scheduler.add_task(
            Task(id=f"t{i}", name=f"Task{i}", priority=5),
            _measure,
        )

    await scheduler.run()

    assert peak <= max_conc


# ---------------------------------------------------------------------------
# 6. Observer events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events_emitted() -> None:
    """on_task_start and on_task_complete events fire for each task."""
    started: list[str] = []
    completed: list[str] = []

    scheduler = TaskScheduler()
    scheduler.on("on_task_start", lambda t: started.append(t.id))
    scheduler.on("on_task_complete", lambda t: completed.append(t.id))

    t = Task(id="obs", name="Observable")
    scheduler.add_task(t, _noop)
    await scheduler.run()

    assert "obs" in started
    assert "obs" in completed


# ---------------------------------------------------------------------------
# 7. Failure propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failure_propagates_to_dependents() -> None:
    """When a task fails, its dependents are also marked FAILED."""
    _fail_counter.clear()

    scheduler = TaskScheduler(base_backoff=0.005)
    t1 = Task(id="root", name="Root", max_retries=0)
    t2 = Task(id="child", name="Child", dependencies=["root"])

    async def _always_fail(task: Task) -> None:
        raise RuntimeError("boom")

    scheduler.add_task(t1, _always_fail)
    scheduler.add_task(t2, _noop)

    await scheduler.run()

    assert t1.status == TaskStatus.FAILED
    assert t2.status == TaskStatus.FAILED
