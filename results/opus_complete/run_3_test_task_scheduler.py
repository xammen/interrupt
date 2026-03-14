"""Tests for the async task scheduler."""

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


# ── Helpers ──────────────────────────────────────────────────────────


async def noop_coro(task: Task) -> str:
    """A coroutine that succeeds immediately."""
    return f"{task.name}-done"


async def slow_coro(task: Task) -> str:
    """A coroutine that takes a short nap."""
    await asyncio.sleep(0.05)
    return f"{task.name}-done"


def _make_fail_coro(fail_times: int):
    """Return a coroutine factory that fails *fail_times* before succeeding."""
    call_count = {"n": 0}

    async def coro(task: Task) -> str:
        call_count["n"] += 1
        if call_count["n"] <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count['n']}")
        return f"{task.name}-recovered"

    return coro


# ── 1. Basic task execution ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_execution() -> None:
    """Tasks with no dependencies should execute and complete."""
    scheduler = TaskScheduler(max_concurrency=2)
    scheduler.add_task(Task(id="a", name="Alpha"), coro=noop_coro)
    scheduler.add_task(Task(id="b", name="Bravo"), coro=noop_coro)

    metrics = await scheduler.run()

    assert scheduler.get_task("a").status == TaskStatus.COMPLETED
    assert scheduler.get_task("b").status == TaskStatus.COMPLETED
    assert scheduler.get_task("a").result == "Alpha-done"
    assert scheduler.get_task("b").result == "Bravo-done"
    assert metrics.total_time >= 0
    assert "a" in metrics.per_task_time
    assert "b" in metrics.per_task_time


# ── 2. Dependency resolution ────────────────────────────────────────


@pytest.mark.asyncio
async def test_dependency_resolution() -> None:
    """Tasks should execute only after their dependencies complete.

    Graph::

        a ──► b ──► d
              │
        c ────┘

    Execution groups: [a, c] then [b] then [d].
    """
    execution_order: list[str] = []

    async def tracking_coro(task: Task) -> str:
        execution_order.append(task.id)
        return task.id

    scheduler = TaskScheduler(max_concurrency=4)
    scheduler.add_task(Task(id="a", name="A"), coro=tracking_coro)
    scheduler.add_task(Task(id="c", name="C"), coro=tracking_coro)
    scheduler.add_task(Task(id="b", name="B", dependencies=["a", "c"]), coro=tracking_coro)
    scheduler.add_task(Task(id="d", name="D", dependencies=["b"]), coro=tracking_coro)

    plan = scheduler.get_execution_plan()

    # a and c should be in the first group (no deps); b second; d third.
    first_group = set(plan[0])
    assert first_group == {"a", "c"}
    assert plan[1] == ["b"]
    assert plan[2] == ["d"]

    await scheduler.run()

    # 'a' and 'c' must appear before 'b'; 'b' before 'd'.
    assert execution_order.index("b") > execution_order.index("a")
    assert execution_order.index("b") > execution_order.index("c")
    assert execution_order.index("d") > execution_order.index("b")


# ── 3. Circular dependency detection ────────────────────────────────


def test_circular_dependency_detection() -> None:
    """A cycle in the dependency graph must raise CircularDependencyError."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="x", name="X", dependencies=["z"]), coro=noop_coro)
    scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), coro=noop_coro)
    scheduler.add_task(Task(id="z", name="Z", dependencies=["y"]), coro=noop_coro)

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


def test_circular_dependency_self_loop() -> None:
    """A task that depends on itself is also a circular dependency."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="s", name="Self", dependencies=["s"]), coro=noop_coro)

    with pytest.raises(CircularDependencyError):
        scheduler.get_execution_plan()


# ── 4. Retry logic with exponential backoff ─────────────────────────


@pytest.mark.asyncio
async def test_retry_with_backoff() -> None:
    """A task that fails twice then succeeds should end as COMPLETED with retry_count=2.

    Uses max_retries=3 and a short backoff — the test asserts that backoff
    delays are applied (total time >= sum of expected delays).
    """
    scheduler = TaskScheduler(max_concurrency=1)
    fail_twice = _make_fail_coro(fail_times=2)
    scheduler.add_task(
        Task(id="r", name="Retry", max_retries=3),
        coro=fail_twice,
    )

    start = time.monotonic()
    metrics = await scheduler.run()
    elapsed = time.monotonic() - start

    task = scheduler.get_task("r")
    assert task.status == TaskStatus.COMPLETED
    assert task.retry_count == 2
    assert task.result == "Retry-recovered"
    assert metrics.retry_counts["r"] == 2

    # Backoffs: 2^0 = 1s after first fail, 2^1 = 2s after second fail → ≥ 3s total.
    # We use a generous lower-bound to avoid flaky CI.
    assert elapsed >= 2.5


@pytest.mark.asyncio
async def test_retry_exhaustion() -> None:
    """A task that always fails should be marked FAILED after max_retries."""
    scheduler = TaskScheduler(max_concurrency=1)
    always_fail = _make_fail_coro(fail_times=999)
    scheduler.add_task(
        Task(id="f", name="Doomed", max_retries=1),
        coro=always_fail,
    )

    await scheduler.run()

    task = scheduler.get_task("f")
    assert task.status == TaskStatus.FAILED
    assert task.retry_count > 0


# ── 5. Concurrent execution respecting concurrency limits ───────────


@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    """No more than max_concurrency tasks should run simultaneously."""
    max_concurrent = 2
    peak_concurrent = {"value": 0}
    current_concurrent = {"value": 0}
    lock = asyncio.Lock()

    async def tracked_coro(task: Task) -> str:
        async with lock:
            current_concurrent["value"] += 1
            if current_concurrent["value"] > peak_concurrent["value"]:
                peak_concurrent["value"] = current_concurrent["value"]
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrent["value"] -= 1
        return task.name

    scheduler = TaskScheduler(max_concurrency=max_concurrent)
    for i in range(6):
        scheduler.add_task(Task(id=f"t{i}", name=f"Task-{i}"), coro=tracked_coro)

    await scheduler.run()

    assert peak_concurrent["value"] <= max_concurrent
    # Verify all completed.
    for i in range(6):
        assert scheduler.get_task(f"t{i}").status == TaskStatus.COMPLETED


# ── 6. Observer / event emission ────────────────────────────────────


@pytest.mark.asyncio
async def test_observer_events() -> None:
    """on_task_start and on_task_complete events should fire for each task."""
    started: list[str] = []
    completed: list[str] = []

    scheduler = TaskScheduler()
    scheduler.on("on_task_start", lambda t: started.append(t.id))
    scheduler.on("on_task_complete", lambda t: completed.append(t.id))
    scheduler.add_task(Task(id="e1", name="E1"), coro=noop_coro)
    scheduler.add_task(Task(id="e2", name="E2"), coro=noop_coro)

    await scheduler.run()

    assert set(started) == {"e1", "e2"}
    assert set(completed) == {"e1", "e2"}


@pytest.mark.asyncio
async def test_observer_on_fail() -> None:
    """on_task_fail should fire when a task exhausts its retries."""
    failed: list[str] = []

    scheduler = TaskScheduler()
    scheduler.on("on_task_fail", lambda t: failed.append(t.id))
    always_fail = _make_fail_coro(fail_times=999)
    scheduler.add_task(
        Task(id="bad", name="Bad", max_retries=0),
        coro=always_fail,
    )

    await scheduler.run()

    assert "bad" in failed


# ── 7. Edge cases ───────────────────────────────────────────────────


def test_priority_validation() -> None:
    """Task priority must be between 1 and 10."""
    with pytest.raises(ValueError):
        Task(id="p", name="P", priority=0)
    with pytest.raises(ValueError):
        Task(id="p", name="P", priority=11)


def test_duplicate_task_id() -> None:
    """Adding two tasks with the same ID should raise."""
    scheduler = TaskScheduler()
    scheduler.add_task(Task(id="dup", name="First"), coro=noop_coro)
    with pytest.raises(ValueError):
        scheduler.add_task(Task(id="dup", name="Second"), coro=noop_coro)
