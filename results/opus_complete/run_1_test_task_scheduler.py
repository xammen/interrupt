"""
Tests for the async task scheduler module.

Run with:  pytest test_task_scheduler.py -v
"""

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

async def _noop(task: Task) -> str:
    """A trivial coroutine that succeeds immediately."""
    return f"{task.name}-done"


async def _failing(task: Task) -> None:
    """A coroutine that always raises."""
    raise RuntimeError(f"{task.name} failed")


def _make_scheduler(*tasks: Task, coro=_noop, max_concurrency: int = 4) -> TaskScheduler:
    """Convenience: create a scheduler and register the given tasks."""
    sched = TaskScheduler(max_concurrency=max_concurrency)
    for t in tasks:
        sched.add_task(t, coro)
    return sched


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_execution() -> None:
    """Tasks with no dependencies should execute and produce results."""
    t1 = Task(id="a", name="Alpha", priority=5)
    t2 = Task(id="b", name="Beta", priority=8)
    sched = _make_scheduler(t1, t2)

    metrics = await sched.run()

    assert sched.get_task("a").status == TaskStatus.COMPLETED
    assert sched.get_task("b").status == TaskStatus.COMPLETED
    assert sched.get_task("a").result == "Alpha-done"
    assert sched.get_task("b").result == "Beta-done"
    assert metrics.total_time > 0
    assert "a" in metrics.task_metrics
    assert "b" in metrics.task_metrics


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks must not start until their dependencies have completed.

    Graph:  C -> B -> A   (C depends on B, B depends on A)
    """
    execution_order: list[str] = []

    async def _track(task: Task) -> str:
        execution_order.append(task.id)
        return f"{task.name}-done"

    t_a = Task(id="a", name="A")
    t_b = Task(id="b", name="B", dependencies=["a"])
    t_c = Task(id="c", name="C", dependencies=["b"])

    sched = TaskScheduler(max_concurrency=1)
    for t in (t_a, t_b, t_c):
        sched.add_task(t, _track)

    await sched.run()

    # With concurrency=1, the execution must be strictly sequential
    assert execution_order == ["a", "b", "c"]
    assert all(sched.get_task(tid).status == TaskStatus.COMPLETED for tid in ("a", "b", "c"))


@pytest.mark.asyncio
async def test_execution_plan_groups() -> None:
    """get_execution_plan returns correct dependency tiers.

    Graph:
        A (no deps)
        B (no deps)
        C depends on A, B
        D depends on C

    Expected groups: [[A, B], [C], [D]]
    """
    t_a = Task(id="a", name="A", priority=3)
    t_b = Task(id="b", name="B", priority=7)
    t_c = Task(id="c", name="C", dependencies=["a", "b"])
    t_d = Task(id="d", name="D", dependencies=["c"])

    sched = _make_scheduler(t_a, t_b, t_c, t_d)
    plan = sched.get_execution_plan()

    assert len(plan) == 3
    group0_ids = {t.id for t in plan[0]}
    assert group0_ids == {"a", "b"}
    assert [t.id for t in plan[1]] == ["c"]
    assert [t.id for t in plan[2]] == ["d"]

    # Within group 0, higher priority should come first
    assert plan[0][0].id == "b"  # priority 7 > 3


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

def test_circular_dependency_detection() -> None:
    """A cycle in the dependency graph must raise CircularDependencyError."""
    t_a = Task(id="a", name="A", dependencies=["b"])
    t_b = Task(id="b", name="B", dependencies=["a"])

    sched = _make_scheduler(t_a, t_b)

    with pytest.raises(CircularDependencyError):
        sched.get_execution_plan()


def test_circular_dependency_three_node() -> None:
    """Detect a cycle spanning three nodes: A -> B -> C -> A."""
    t_a = Task(id="a", name="A", dependencies=["c"])
    t_b = Task(id="b", name="B", dependencies=["a"])
    t_c = Task(id="c", name="C", dependencies=["b"])

    sched = _make_scheduler(t_a, t_b, t_c)

    with pytest.raises(CircularDependencyError):
        sched.get_execution_plan()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff() -> None:
    """A failing task retries up to max_retries with increasing delays.

    We use max_retries=2 and verify that:
      - The task is retried exactly 2 extra times (3 attempts total).
      - Total elapsed time is >= 1 + 2 = 3 seconds of backoff.
      - The task ends with FAILED status.
    """
    call_count = 0

    async def _fail_always(task: Task) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    task = Task(id="x", name="X", max_retries=2)
    sched = TaskScheduler()
    sched.add_task(task, _fail_always)

    start = time.monotonic()
    metrics = await sched.run()
    elapsed = time.monotonic() - start

    assert sched.get_task("x").status == TaskStatus.FAILED
    assert call_count == 3  # initial + 2 retries
    assert sched.get_task("x").retry_count == 3  # incremented past max
    # Backoff was 1s + 2s = 3s minimum
    assert elapsed >= 2.5  # allow small timing tolerance
    assert metrics.task_metrics["x"].retries == 3


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt() -> None:
    """Task fails once then succeeds — should end COMPLETED with retry_count=1."""
    attempt = 0

    async def _fail_then_succeed(task: Task) -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("transient")
        return "ok"

    task = Task(id="y", name="Y", max_retries=3)
    sched = TaskScheduler()
    sched.add_task(task, _fail_then_succeed)
    await sched.run()

    assert sched.get_task("y").status == TaskStatus.COMPLETED
    assert sched.get_task("y").result == "ok"
    assert sched.get_task("y").retry_count == 1


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    """No more than max_concurrency tasks should run at the same time."""
    max_concurrent = 2
    active: int = 0
    peak: int = 0

    async def _track_concurrency(task: Task) -> str:
        nonlocal active, peak
        active += 1
        if active > peak:
            peak = active
        await asyncio.sleep(0.05)  # simulate work
        active -= 1
        return "done"

    tasks = [Task(id=str(i), name=f"T{i}") for i in range(6)]
    sched = TaskScheduler(max_concurrency=max_concurrent)
    for t in tasks:
        sched.add_task(t, _track_concurrency)

    await sched.run()

    assert peak <= max_concurrent
    assert all(
        sched.get_task(str(i)).status == TaskStatus.COMPLETED for i in range(6)
    )


# ---------------------------------------------------------------------------
# 6. Observer / event callbacks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events() -> None:
    """on_task_start, on_task_complete, and on_task_fail events fire."""
    started: list[str] = []
    completed: list[str] = []
    failed: list[str] = []

    async def _fail(task: Task) -> None:
        raise RuntimeError("oops")

    t_ok = Task(id="ok", name="OK", max_retries=0)
    t_bad = Task(id="bad", name="BAD", max_retries=0)

    sched = TaskScheduler()
    sched.add_task(t_ok, _noop)
    sched.add_task(t_bad, _fail)

    sched.on("on_task_start", lambda t: started.append(t.id))
    sched.on("on_task_complete", lambda t: completed.append(t.id))
    sched.on("on_task_fail", lambda t, exc: failed.append(t.id))

    await sched.run()

    assert "ok" in started
    assert "bad" in started
    assert "ok" in completed
    assert "bad" in failed
    assert "bad" not in completed


# ---------------------------------------------------------------------------
# 7. Edge-case: dependency on a failed task
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependent_fails_when_dependency_fails() -> None:
    """If a dependency fails, its dependents should also be marked FAILED."""

    async def _fail(task: Task) -> None:
        raise RuntimeError("upstream failure")

    t_a = Task(id="a", name="A", max_retries=0)
    t_b = Task(id="b", name="B", dependencies=["a"])

    sched = TaskScheduler()
    sched.add_task(t_a, _fail)
    sched.add_task(t_b, _noop)

    await sched.run()

    assert sched.get_task("a").status == TaskStatus.FAILED
    assert sched.get_task("b").status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 8. Validation: unknown dependency
# ---------------------------------------------------------------------------

def test_unknown_dependency_raises() -> None:
    """Referencing a non-existent task ID in dependencies raises ValueError."""
    task = Task(id="a", name="A", dependencies=["nonexistent"])
    sched = TaskScheduler()
    sched.add_task(task, _noop)

    with pytest.raises(ValueError, match="unknown task"):
        sched.get_execution_plan()


# ---------------------------------------------------------------------------
# 9. Validation: priority bounds
# ---------------------------------------------------------------------------

def test_priority_out_of_range() -> None:
    """Task priority must be between 1 and 10 inclusive."""
    with pytest.raises(ValueError, match="priority"):
        Task(id="z", name="Z", priority=0)
    with pytest.raises(ValueError, match="priority"):
        Task(id="z", name="Z", priority=11)
