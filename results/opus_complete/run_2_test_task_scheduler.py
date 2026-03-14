"""Tests for the async task scheduler."""

from __future__ import annotations

import asyncio
import time
from typing import Any, List

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskNotFoundError,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_coro(task: Task) -> str:
    """A trivial coroutine that succeeds immediately."""
    return f"{task.name}-done"


async def _slow_coro(task: Task) -> str:
    """A coroutine that takes a short while."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


def _make_failing_coro(fail_times: int = 1):
    """Return a coroutine factory that fails *fail_times* then succeeds."""
    call_count = {"n": 0}

    async def _coro(task: Task) -> str:
        call_count["n"] += 1
        if call_count["n"] <= fail_times:
            raise RuntimeError(f"Transient error attempt {call_count['n']}")
        return f"{task.name}-recovered"

    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_task_execution() -> None:
    """Tasks without dependencies should execute and complete."""
    scheduler = TaskScheduler()
    t1 = Task(id="a", name="alpha", priority=5)
    t2 = Task(id="b", name="beta", priority=8)

    scheduler.add_task(t1, _noop_coro)
    scheduler.add_task(t2, _noop_coro)

    results = await scheduler.run()

    assert results["a"].status == TaskStatus.COMPLETED
    assert results["a"].result == "alpha-done"
    assert results["b"].status == TaskStatus.COMPLETED
    assert results["b"].result == "beta-done"

    # Metrics should have been recorded
    assert scheduler.metrics.total_time is not None
    assert scheduler.metrics.total_time >= 0
    for tid in ("a", "b"):
        m = scheduler.metrics.task_metrics[tid]
        assert m.duration is not None
        assert m.duration >= 0


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks should execute in dependency order (execution plan groups)."""
    scheduler = TaskScheduler()

    # c depends on a and b; d depends on c
    scheduler.add_task(Task(id="a", name="A"), _noop_coro)
    scheduler.add_task(Task(id="b", name="B"), _noop_coro)
    scheduler.add_task(Task(id="c", name="C", dependencies=["a", "b"]), _noop_coro)
    scheduler.add_task(Task(id="d", name="D", dependencies=["c"]), _noop_coro)

    plan = scheduler.get_execution_plan()

    # a and b should be in the first group, c in the second, d in the third
    assert len(plan) == 3
    assert set(plan[0]) == {"a", "b"}
    assert plan[1] == ["c"]
    assert plan[2] == ["d"]

    results = await scheduler.run()
    assert all(t.status == TaskStatus.COMPLETED for t in results.values())


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------


def test_circular_dependency_detection() -> None:
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = TaskScheduler()

    scheduler.add_task(Task(id="x", name="X", dependencies=["z"]), _noop_coro)
    scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop_coro)
    scheduler.add_task(Task(id="z", name="Z", dependencies=["y"]), _noop_coro)

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


def test_circular_dependency_self_loop() -> None:
    """A task depending on itself must also be detected."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="s", name="Self", dependencies=["s"]), _noop_coro)

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_with_backoff() -> None:
    """A task that fails transiently should retry and eventually succeed."""
    scheduler = TaskScheduler()
    # Fail twice, succeed on third attempt. max_retries=3 allows this.
    coro = _make_failing_coro(fail_times=2)
    task = Task(id="r", name="Retryable", max_retries=3)
    scheduler.add_task(task, coro)

    start = time.monotonic()
    results = await scheduler.run()
    elapsed = time.monotonic() - start

    assert results["r"].status == TaskStatus.COMPLETED
    assert results["r"].result == "Retryable-recovered"
    assert results["r"].retry_count == 2
    # Backoff: 0.1s + 0.2s = 0.3s minimum
    assert elapsed >= 0.25  # Allow a small margin


@pytest.mark.asyncio
async def test_retry_exhausted() -> None:
    """A task that always fails should be marked FAILED after max retries."""
    scheduler = TaskScheduler()

    async def _always_fail(task: Task) -> None:
        raise RuntimeError("permanent failure")

    task = Task(id="f", name="AlwaysFails", max_retries=2)
    scheduler.add_task(task, _always_fail)

    fail_events: List[Task] = []
    scheduler.on("on_task_fail", lambda t, exc: fail_events.append(t))

    results = await scheduler.run()

    assert results["f"].status == TaskStatus.FAILED
    assert isinstance(results["f"].result, RuntimeError)
    # retry_count should be max_retries + 1 (the final failing attempt)
    assert results["f"].retry_count == 3  # attempts: 1, 2, 3
    assert len(fail_events) == 1


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    """No more than max_concurrency tasks should run simultaneously."""
    max_conc = 2
    scheduler = TaskScheduler(max_concurrency=max_conc)

    peak_concurrency = {"value": 0}
    current_concurrency = {"value": 0}

    async def _track_concurrency(task: Task) -> str:
        current_concurrency["value"] += 1
        if current_concurrency["value"] > peak_concurrency["value"]:
            peak_concurrency["value"] = current_concurrency["value"]
        await asyncio.sleep(0.05)
        current_concurrency["value"] -= 1
        return f"{task.name}-done"

    for i in range(6):
        scheduler.add_task(Task(id=f"t{i}", name=f"Task{i}"), _track_concurrency)

    await scheduler.run()

    assert peak_concurrency["value"] <= max_conc


# ---------------------------------------------------------------------------
# 6. Observer pattern events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_events() -> None:
    """on_task_start and on_task_complete should fire for each task."""
    scheduler = TaskScheduler()
    started: List[str] = []
    completed: List[str] = []

    scheduler.on("on_task_start", lambda t: started.append(t.id))
    scheduler.on("on_task_complete", lambda t: completed.append(t.id))

    scheduler.add_task(Task(id="e1", name="E1"), _noop_coro)
    scheduler.add_task(Task(id="e2", name="E2"), _noop_coro)

    await scheduler.run()

    assert set(started) == {"e1", "e2"}
    assert set(completed) == {"e1", "e2"}


# ---------------------------------------------------------------------------
# 7. Missing dependency raises TaskNotFoundError
# ---------------------------------------------------------------------------


def test_missing_dependency() -> None:
    """Referencing a non-existent dependency must raise TaskNotFoundError."""
    scheduler = TaskScheduler()
    scheduler.add_task(
        Task(id="orphan", name="Orphan", dependencies=["ghost"]), _noop_coro
    )

    with pytest.raises(TaskNotFoundError):
        scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 8. Cascade failure on dependency failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_failure() -> None:
    """If a dependency fails, downstream tasks should also be marked FAILED."""
    scheduler = TaskScheduler()

    async def _fail(task: Task) -> None:
        raise RuntimeError("boom")

    scheduler.add_task(Task(id="root", name="Root", max_retries=0), _fail)
    scheduler.add_task(
        Task(id="child", name="Child", dependencies=["root"]), _noop_coro
    )

    results = await scheduler.run()

    assert results["root"].status == TaskStatus.FAILED
    assert results["child"].status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 9. Priority ordering within a group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_ordering_in_plan() -> None:
    """Higher-priority tasks should appear first within an execution group."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="lo", name="Low", priority=1), _noop_coro)
    scheduler.add_task(Task(id="hi", name="High", priority=10), _noop_coro)
    scheduler.add_task(Task(id="mid", name="Mid", priority=5), _noop_coro)

    plan = scheduler.get_execution_plan()

    # All three are independent so they share a single group
    assert len(plan) == 1
    assert plan[0] == ["hi", "mid", "lo"]
