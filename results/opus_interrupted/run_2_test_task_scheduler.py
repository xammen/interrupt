"""
Unit tests for the async task scheduler.
"""

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

def _make_scheduler(**kwargs) -> TaskScheduler:
    """Shortcut to build a scheduler with sensible test defaults."""
    return TaskScheduler(base_backoff=0.01, **kwargs)


async def _noop_executor(task: Task) -> str:
    """Executor that instantly succeeds."""
    return f"{task.name}-done"


async def _slow_executor(task: Task) -> str:
    """Executor that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution() -> None:
    """Tasks with no dependencies should run and complete successfully."""
    scheduler = _make_scheduler()
    t1 = Task(id="a", name="alpha")
    t2 = Task(id="b", name="bravo")

    scheduler.add_task(t1, _noop_executor)
    scheduler.add_task(t2, _noop_executor)

    metrics = await scheduler.run()

    assert t1.status == TaskStatus.COMPLETED
    assert t2.status == TaskStatus.COMPLETED
    assert t1.result == "alpha-done"
    assert t2.result == "bravo-done"
    assert metrics.total_time > 0
    assert "a" in metrics.per_task_time
    assert "b" in metrics.per_task_time


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks should execute only after their dependencies are complete.

    Graph:  a -> b -> c   (c depends on b, b depends on a)
    Expected plan: [[a], [b], [c]]
    """
    scheduler = _make_scheduler()
    execution_order: list[str] = []

    async def _tracking_executor(task: Task) -> str:
        execution_order.append(task.id)
        return task.name

    scheduler.add_task(Task(id="a", name="A"), _tracking_executor)
    scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _tracking_executor)
    scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _tracking_executor)

    await scheduler.run()

    assert execution_order == ["a", "b", "c"]

    plan = scheduler.get_execution_plan()
    assert plan == [["a"], ["b"], ["c"]]


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circular_dependency_detection() -> None:
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = _make_scheduler()

    scheduler.add_task(Task(id="x", name="X", dependencies=["z"]), _noop_executor)
    scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop_executor)
    scheduler.add_task(Task(id="z", name="Z", dependencies=["y"]), _noop_executor)

    with pytest.raises(CircularDependencyError):
        await scheduler.run()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff() -> None:
    """A task that fails twice then succeeds should be retried with backoff."""
    scheduler = _make_scheduler()  # base_backoff=0.01
    call_count = 0

    async def _flaky_executor(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("transient failure")
        return "ok"

    t = Task(id="flaky", name="Flaky", max_retries=3)
    scheduler.add_task(t, _flaky_executor)

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    assert t.status == TaskStatus.COMPLETED
    assert t.result == "ok"
    assert call_count == 3  # 1 initial + 2 retries
    assert t.retry_count == 2
    assert metrics.retry_counts["flaky"] == 2

    # Exponential backoff: 0.01 + 0.02 = 0.03s minimum
    assert elapsed >= 0.02


@pytest.mark.asyncio
async def test_retry_exhaustion() -> None:
    """A task that always fails should exhaust retries and be marked FAILED."""
    scheduler = _make_scheduler()

    async def _always_fail(task: Task) -> None:
        raise RuntimeError("permanent failure")

    t = Task(id="doom", name="Doom", max_retries=2)
    scheduler.add_task(t, _always_fail)

    await scheduler.run()

    assert t.status == TaskStatus.FAILED
    assert isinstance(t.result, RuntimeError)
    assert t.retry_count == 3  # initial try + 2 retries = 3 increments > max_retries=2


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    """No more than max_concurrency tasks should run at the same time."""
    max_conc = 2
    scheduler = _make_scheduler(max_concurrency=max_conc)
    peak_concurrency = 0
    current_concurrency = 0
    lock = asyncio.Lock()

    async def _counting_executor(task: Task) -> str:
        nonlocal peak_concurrency, current_concurrency
        async with lock:
            current_concurrency += 1
            if current_concurrency > peak_concurrency:
                peak_concurrency = current_concurrency
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrency -= 1
        return "done"

    # 5 independent tasks, max concurrency = 2
    for i in range(5):
        scheduler.add_task(Task(id=f"t{i}", name=f"Task{i}"), _counting_executor)

    await scheduler.run()

    assert peak_concurrency <= max_conc
    # With 5 tasks and concurrency 2, peak should actually reach 2.
    assert peak_concurrency == max_conc


# ---------------------------------------------------------------------------
# 6. Observer / event emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_emission() -> None:
    """Observer callbacks should fire for start, complete, and fail events."""
    scheduler = _make_scheduler()
    events: list[tuple[str, str]] = []

    scheduler.on("on_task_start", lambda t: events.append(("start", t.id)))
    scheduler.on("on_task_complete", lambda t: events.append(("complete", t.id)))
    scheduler.on("on_task_fail", lambda t: events.append(("fail", t.id)))

    scheduler.add_task(Task(id="good", name="Good"), _noop_executor)

    async def _bad(task: Task) -> None:
        raise RuntimeError("boom")

    scheduler.add_task(Task(id="bad", name="Bad", max_retries=0), _bad)

    await scheduler.run()

    # "good" should have start + complete
    assert ("start", "good") in events
    assert ("complete", "good") in events

    # "bad" should have start + fail
    assert ("start", "bad") in events
    assert ("fail", "bad") in events


# ---------------------------------------------------------------------------
# 7. Execution plan with priorities
# ---------------------------------------------------------------------------

def test_execution_plan_priority_ordering() -> None:
    """Within an execution group, higher-priority tasks should come first."""
    scheduler = _make_scheduler()
    scheduler.add_task(Task(id="lo", name="Low", priority=1), _noop_executor)
    scheduler.add_task(Task(id="hi", name="High", priority=10), _noop_executor)
    scheduler.add_task(Task(id="mid", name="Mid", priority=5), _noop_executor)

    plan = scheduler.get_execution_plan()

    # All independent -> single group, ordered by descending priority.
    assert len(plan) == 1
    assert plan[0] == ["hi", "mid", "lo"]


# ---------------------------------------------------------------------------
# 8. Task validation
# ---------------------------------------------------------------------------

def test_task_priority_validation() -> None:
    """Task priority outside 1-10 should raise ValueError."""
    with pytest.raises(ValueError):
        Task(id="x", name="X", priority=0)
    with pytest.raises(ValueError):
        Task(id="y", name="Y", priority=11)
