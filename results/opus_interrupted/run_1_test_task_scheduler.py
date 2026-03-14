"""Tests for the async task scheduler."""

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

async def _noop_coro(task: Task) -> str:
    """A trivial coroutine that returns immediately."""
    return f"{task.name}-done"


async def _slow_coro(task: Task) -> str:
    """A coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


def _make_failing_coro(fail_times: int):
    """Return a coroutine factory that fails *fail_times* before succeeding."""
    call_count = 0

    async def coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"Transient failure #{call_count}")
        return f"{task.name}-recovered"

    return coro


def _make_always_failing_coro():
    """Return a coroutine that always raises."""
    async def coro(task: Task) -> str:
        raise RuntimeError("Permanent failure")
    return coro


# ---------------------------------------------------------------------------
# Test 1 – Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution():
    """A single task with no dependencies should execute and produce a result."""
    scheduler = TaskScheduler()
    task = Task(id="t1", name="hello", priority=5)
    scheduler.add_task(task, _noop_coro)

    results = await scheduler.run()

    assert results["t1"] == "hello-done"
    assert task.status == TaskStatus.COMPLETED
    assert "t1" in scheduler.metrics.task_metrics
    assert scheduler.metrics.total_time > 0


# ---------------------------------------------------------------------------
# Test 2 – Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution():
    """Tasks must run after their dependencies complete.

    Graph:  t1 -> t2 -> t3  (t3 depends on t2, t2 depends on t1)
    Expected execution groups: [[t1], [t2], [t3]]
    """
    scheduler = TaskScheduler()

    t1 = Task(id="t1", name="first")
    t2 = Task(id="t2", name="second", dependencies=["t1"])
    t3 = Task(id="t3", name="third", dependencies=["t2"])

    execution_order: list[str] = []

    async def tracking_coro(task: Task) -> str:
        execution_order.append(task.id)
        return f"{task.name}-done"

    scheduler.add_task(t1, tracking_coro)
    scheduler.add_task(t2, tracking_coro)
    scheduler.add_task(t3, tracking_coro)

    plan = scheduler.get_execution_plan()
    assert plan == [["t1"], ["t2"], ["t3"]]

    results = await scheduler.run()

    assert execution_order == ["t1", "t2", "t3"]
    assert all(v.endswith("-done") for v in results.values())


# ---------------------------------------------------------------------------
# Test 3 – Circular dependency detection
# ---------------------------------------------------------------------------

def test_circular_dependency_detection():
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = TaskScheduler()

    t1 = Task(id="t1", name="a", dependencies=["t3"])
    t2 = Task(id="t2", name="b", dependencies=["t1"])
    t3 = Task(id="t3", name="c", dependencies=["t2"])

    scheduler.add_task(t1, _noop_coro)
    scheduler.add_task(t2, _noop_coro)
    scheduler.add_task(t3, _noop_coro)

    with pytest.raises(CircularDependencyError) as exc_info:
        scheduler.get_execution_plan()

    assert len(exc_info.value.cycle) >= 3


# ---------------------------------------------------------------------------
# Test 4 – Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_with_exponential_backoff():
    """A task that fails twice then succeeds should complete after 2 retries.

    Backoff schedule: retry 1 -> 0.1 s, retry 2 -> 0.2 s.
    Total extra delay ≈ 0.3 s.
    """
    scheduler = TaskScheduler()
    coro = _make_failing_coro(fail_times=2)
    task = Task(id="t1", name="flaky", max_retries=3)
    scheduler.add_task(task, coro)

    start = time.monotonic()
    results = await scheduler.run()
    elapsed = time.monotonic() - start

    assert results["t1"] == "flaky-recovered"
    assert task.status == TaskStatus.COMPLETED
    assert task.retry_count == 2
    # Verify that backoff actually introduced a measurable delay.
    assert elapsed >= 0.25  # 0.1 + 0.2 = 0.3, allow some tolerance


@pytest.mark.asyncio
async def test_retry_exhaustion():
    """A task that always fails should end with FAILED after max_retries."""
    scheduler = TaskScheduler()
    coro = _make_always_failing_coro()
    task = Task(id="t1", name="doomed", max_retries=2)

    fail_events: list[str] = []
    scheduler.on_task_fail(lambda t: fail_events.append(t.id))

    scheduler.add_task(task, coro)
    results = await scheduler.run()

    assert results["t1"] is None
    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 3  # 0 + 3 increments > max_retries=2
    assert "t1" in fail_events


# ---------------------------------------------------------------------------
# Test 5 – Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit():
    """With concurrency=2 and 4 independent tasks, at most 2 should run at once."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def concurrency_tracking_coro(task: Task) -> str:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        await asyncio.sleep(0.1)  # Simulate work so overlap can happen.
        async with lock:
            current_concurrent -= 1
        return f"{task.name}-done"

    scheduler = TaskScheduler(max_concurrency=2)
    for i in range(4):
        t = Task(id=f"t{i}", name=f"task-{i}", priority=5)
        scheduler.add_task(t, concurrency_tracking_coro)

    results = await scheduler.run()

    assert all(v.endswith("-done") for v in results.values())
    assert max_concurrent <= 2


# ---------------------------------------------------------------------------
# Test 6 – Observer pattern events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events():
    """Start and complete events should fire for every successful task."""
    started: list[str] = []
    completed: list[str] = []

    scheduler = TaskScheduler()
    scheduler.on_task_start(lambda t: started.append(t.id))
    scheduler.on_task_complete(lambda t: completed.append(t.id))

    scheduler.add_task(Task(id="a", name="A"), _noop_coro)
    scheduler.add_task(Task(id="b", name="B"), _noop_coro)

    await scheduler.run()

    assert "a" in started and "b" in started
    assert "a" in completed and "b" in completed


# ---------------------------------------------------------------------------
# Test 7 – Execution plan with parallel groups
# ---------------------------------------------------------------------------

def test_execution_plan_parallel_groups():
    """Independent tasks should be grouped together in the execution plan.

    Graph:
        t1 (no deps)
        t2 (no deps)
        t3 depends on [t1, t2]

    Expected plan: [[t1, t2], [t3]]  (t1 and t2 are independent)
    """
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="t1", name="A", priority=8), _noop_coro)
    scheduler.add_task(Task(id="t2", name="B", priority=6), _noop_coro)
    scheduler.add_task(Task(id="t3", name="C", dependencies=["t1", "t2"]), _noop_coro)

    plan = scheduler.get_execution_plan()

    assert len(plan) == 2
    # First group has the two independent tasks, sorted by priority desc.
    assert plan[0] == ["t1", "t2"]
    assert plan[1] == ["t3"]
