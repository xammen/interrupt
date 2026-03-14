"""
test_task_scheduler.py - pytest test suite for task_scheduler.py
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock

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

def make_task(
    id: str,
    name: str = "",
    priority: int = 5,
    dependencies: List[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=id,
        name=name or id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


def success_fn(value=None):
    async def _fn():
        return value
    return _fn


def failing_fn(exc_type=RuntimeError, message="boom"):
    async def _fn():
        raise exc_type(message)
    return _fn


def counting_fn(counter: list, succeed_after: int = 0):
    """Fails the first *succeed_after* calls, then succeeds."""
    async def _fn():
        counter.append(1)
        if len(counter) <= succeed_after:
            raise RuntimeError("not yet")
        return "ok"
    return _fn


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_task_execution():
    """A single task with no dependencies should complete successfully."""
    scheduler = TaskScheduler()
    task = make_task("t1")
    scheduler.add_task(task, success_fn("hello"))

    await scheduler.run()

    assert task.status == TaskStatus.COMPLETED
    assert task.result == "hello"


@pytest.mark.asyncio
async def test_multiple_independent_tasks_all_complete():
    """Multiple independent tasks should all reach COMPLETED."""
    scheduler = TaskScheduler()
    tasks = [make_task(f"t{i}") for i in range(5)]
    for t in tasks:
        scheduler.add_task(t, success_fn(t.id))

    await scheduler.run()

    for t in tasks:
        assert t.status == TaskStatus.COMPLETED
        assert t.result == t.id


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_resolution_order():
    """Task B must run after Task A when B depends on A."""
    execution_order: List[str] = []

    async def record(name: str):
        execution_order.append(name)

    scheduler = TaskScheduler(max_concurrency=1)

    t_a = make_task("a")
    t_b = make_task("b", dependencies=["a"])

    scheduler.add_task(t_a, lambda: record("a"))
    scheduler.add_task(t_b, lambda: record("b"))

    await scheduler.run()

    assert execution_order.index("a") < execution_order.index("b")
    assert t_a.status == TaskStatus.COMPLETED
    assert t_b.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_execution_plan_groups():
    """get_execution_plan should return correct dependency groups."""
    scheduler = TaskScheduler()

    # Graph: a -> c, b -> c, c (no deps), d -> c
    scheduler.add_task(make_task("a", dependencies=["c"]), success_fn())
    scheduler.add_task(make_task("b", dependencies=["c"]), success_fn())
    scheduler.add_task(make_task("c"), success_fn())
    scheduler.add_task(make_task("d", dependencies=["c"]), success_fn())

    plan = scheduler.get_execution_plan()

    # "c" must appear before "a", "b", "d"
    flat = [tid for group in plan for tid in group]
    c_idx = flat.index("c")
    for tid in ("a", "b", "d"):
        assert flat.index(tid) > c_idx, f"{tid} should come after c"


@pytest.mark.asyncio
async def test_unknown_dependency_raises():
    """Referencing a non-existent dependency should raise TaskNotFoundError."""
    scheduler = TaskScheduler()
    scheduler.add_task(make_task("x", dependencies=["missing"]), success_fn())

    with pytest.raises(TaskNotFoundError):
        scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

def test_circular_dependency_direct():
    """Direct cycle A -> B -> A must raise CircularDependencyError."""
    scheduler = TaskScheduler()
    scheduler.add_task(make_task("a", dependencies=["b"]), success_fn())
    scheduler.add_task(make_task("b", dependencies=["a"]), success_fn())

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


def test_circular_dependency_indirect():
    """Indirect cycle A -> B -> C -> A must raise CircularDependencyError."""
    scheduler = TaskScheduler()
    scheduler.add_task(make_task("a", dependencies=["b"]), success_fn())
    scheduler.add_task(make_task("b", dependencies=["c"]), success_fn())
    scheduler.add_task(make_task("c", dependencies=["a"]), success_fn())

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


def test_no_false_positive_circular_detection():
    """A diamond DAG (a->b, a->c, b->d, c->d) must NOT raise."""
    scheduler = TaskScheduler()
    scheduler.add_task(make_task("a", dependencies=["b", "c"]), success_fn())
    scheduler.add_task(make_task("b", dependencies=["d"]), success_fn())
    scheduler.add_task(make_task("c", dependencies=["d"]), success_fn())
    scheduler.add_task(make_task("d"), success_fn())

    plan = scheduler.get_execution_plan()  # must not raise
    assert plan  # non-empty


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_succeeds_after_failures(monkeypatch):
    """Task should succeed on the third attempt (fails twice first)."""
    sleep_calls: List[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    counter: List[int] = []
    scheduler = TaskScheduler(base_backoff=1.0)
    task = make_task("r1", max_retries=3)
    # succeed_after=2 means fail on calls 1 & 2, succeed on call 3
    scheduler.add_task(task, counting_fn(counter, succeed_after=2))

    await scheduler.run()

    assert task.status == TaskStatus.COMPLETED
    assert task.retry_count == 2
    assert len(counter) == 3

    # Backoff delays: 1*2^0=1, 1*2^1=2
    assert sleep_calls == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_task_fails_after_max_retries(monkeypatch):
    """Task should end as FAILED when all retry attempts are exhausted."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    scheduler = TaskScheduler(base_backoff=0.0)
    task = make_task("r2", max_retries=2)
    scheduler.add_task(task, failing_fn())

    await scheduler.run()

    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 3  # initial attempt + 2 retries
    assert isinstance(task.result, RuntimeError)


@pytest.mark.asyncio
async def test_no_retry_on_zero_max_retries(monkeypatch):
    """With max_retries=0 the task should fail immediately without sleeping."""
    sleep_calls: List[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    scheduler = TaskScheduler()
    task = make_task("r3", max_retries=0)
    scheduler.add_task(task, failing_fn())

    await scheduler.run()

    assert task.status == TaskStatus.FAILED
    assert sleep_calls == []  # no backoff sleeps


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit_respected():
    """No more than max_concurrency tasks should run simultaneously."""
    max_concurrency = 2
    active: List[int] = []
    peak: List[int] = []

    async def tracked_fn():
        active.append(1)
        peak.append(len(active))
        await asyncio.sleep(0.05)  # simulate work
        active.pop()

    scheduler = TaskScheduler(max_concurrency=max_concurrency)
    for i in range(6):
        t = make_task(f"c{i}", priority=5)
        scheduler.add_task(t, tracked_fn)

    await scheduler.run()

    assert max(peak) <= max_concurrency, (
        f"Peak concurrency {max(peak)} exceeded limit {max_concurrency}"
    )


@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently():
    """Independent tasks in the same group should overlap in execution time."""
    started: List[float] = []
    ended: List[float] = []

    async def timed_fn():
        started.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.1)
        ended.append(asyncio.get_event_loop().time())

    scheduler = TaskScheduler(max_concurrency=4)
    for i in range(3):
        scheduler.add_task(make_task(f"p{i}"), timed_fn)

    t0 = asyncio.get_event_loop().time()
    await scheduler.run()
    elapsed = asyncio.get_event_loop().time() - t0

    # If truly concurrent, total time should be ~0.1 s, not ~0.3 s
    assert elapsed < 0.25, f"Tasks do not appear to be running concurrently (elapsed={elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# 6. Observer / event callbacks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_events_fired():
    """on_task_start and on_task_complete callbacks must be invoked."""
    started_ids: List[str] = []
    completed_ids: List[str] = []
    failed_ids: List[str] = []

    async def on_start(task: Task) -> None:
        started_ids.append(task.id)

    async def on_complete(task: Task) -> None:
        completed_ids.append(task.id)

    async def on_fail(task: Task) -> None:
        failed_ids.append(task.id)

    scheduler = TaskScheduler()
    scheduler.on_task_start(on_start)
    scheduler.on_task_complete(on_complete)
    scheduler.on_task_fail(on_fail)

    scheduler.add_task(make_task("ok"), success_fn())
    scheduler.add_task(make_task("bad", max_retries=0), failing_fn())

    await scheduler.run()

    assert "ok" in started_ids
    assert "ok" in completed_ids
    assert "bad" in failed_ids
    assert "ok" not in failed_ids


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_collected():
    """Metrics should capture per-task elapsed time and total time."""
    scheduler = TaskScheduler()
    scheduler.add_task(make_task("m1"), success_fn())
    scheduler.add_task(make_task("m2"), success_fn())

    metrics = await scheduler.run()

    for tid in ("m1", "m2"):
        assert metrics[tid].elapsed is not None
        assert metrics[tid].elapsed >= 0

    assert "__total__" in metrics
    assert metrics["__total__"].elapsed is not None
    assert metrics["__total__"].elapsed >= 0


# ---------------------------------------------------------------------------
# 8. Priority ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_priority_ordering_within_group(monkeypatch):
    """Higher-priority tasks should be submitted first within an execution group."""
    order: List[str] = []

    # Patch Semaphore to serialise execution so order is deterministic
    scheduler = TaskScheduler(max_concurrency=1)

    async def track(name: str):
        order.append(name)

    scheduler.add_task(make_task("low", priority=1), lambda: track("low"))
    scheduler.add_task(make_task("mid", priority=5), lambda: track("mid"))
    scheduler.add_task(make_task("high", priority=10), lambda: track("high"))

    await scheduler.run()

    assert order.index("high") < order.index("mid")
    assert order.index("mid") < order.index("low")
