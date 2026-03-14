"""
test_task_scheduler.py - Pytest suite for task_scheduler.py

Covers:
  1. Basic task execution
  2. Dependency resolution (topological ordering)
  3. Circular dependency detection
  4. Retry logic with exponential backoff
  5. Concurrent execution respecting the concurrency limit
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

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
    task_id: str,
    name: str = "",
    priority: int = 5,
    dependencies: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def succeed(value: object = "ok"):
    """Coroutine factory that returns *value* immediately."""
    return value


def always_succeed(value: object = "ok"):
    """Return a fresh coroutine each time it is called."""
    async def _inner():
        return value
    return _inner


def always_fail(exc: Exception | None = None):
    """Return a coroutine that always raises *exc*."""
    async def _inner():
        raise (exc or RuntimeError("boom"))
    return _inner


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    """A single task with no dependencies should complete successfully."""

    def test_task_result_stored(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, always_succeed("hello"))

        results = asyncio.run(scheduler.run())

        assert results["t1"] == "hello"
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "hello"

    def test_metrics_populated(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, always_succeed())

        asyncio.run(scheduler.run())

        m = scheduler.metrics
        assert m.total_time > 0
        assert "t1" in m.task_metrics
        assert m.task_metrics["t1"].elapsed is not None
        assert m.task_metrics["t1"].elapsed >= 0

    def test_observer_events_fired(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, always_succeed())

        started: list[str] = []
        completed: list[str] = []

        scheduler.on("on_task_start", lambda task: started.append(task.id))
        scheduler.on("on_task_complete", lambda task: completed.append(task.id))

        asyncio.run(scheduler.run())

        assert started == ["t1"]
        assert completed == ["t1"]

    def test_multiple_independent_tasks_all_complete(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(make_task(f"t{i}"), always_succeed(i))

        results = asyncio.run(scheduler.run())

        for i in range(5):
            assert results[f"t{i}"] == i


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    """Tasks with dependencies should execute in the correct order."""

    def test_linear_chain_executes_in_order(self):
        """t1 -> t2 -> t3 must run in that sequence."""
        execution_order: list[str] = []

        def make_recording_coro(tid: str):
            async def _inner():
                execution_order.append(tid)
                return tid
            return _inner

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), make_recording_coro("t1"))
        scheduler.add_task(
            make_task("t2", dependencies=["t1"]), make_recording_coro("t2")
        )
        scheduler.add_task(
            make_task("t3", dependencies=["t2"]), make_recording_coro("t3")
        )

        asyncio.run(scheduler.run())

        assert execution_order == ["t1", "t2", "t3"]

    def test_get_execution_plan_groups(self):
        """
        Graph: A  B -> C -> D
        Expected groups: [A, B] then [C] then [D]
        """
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("A"), always_succeed())
        scheduler.add_task(make_task("B"), always_succeed())
        scheduler.add_task(
            make_task("C", dependencies=["B"]), always_succeed()
        )
        scheduler.add_task(
            make_task("D", dependencies=["C"]), always_succeed()
        )

        plan = scheduler.get_execution_plan()

        # Flatten to check overall ordering guarantees
        flat = [tid for group in plan for tid in group]
        assert flat.index("B") < flat.index("C") < flat.index("D")
        # A and B must be in the same first group
        assert "A" in plan[0] and "B" in plan[0]

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            make_task("t1", dependencies=["ghost"]), always_succeed()
        )

        with pytest.raises(TaskNotFoundError):
            scheduler.get_execution_plan()

    def test_diamond_dependency_resolved(self):
        """
        A -> B, A -> C, B -> D, C -> D  (diamond shape)
        D should run last.
        """
        order: list[str] = []

        def rec(tid: str):
            async def _inner():
                order.append(tid)
                return tid
            return _inner

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("A"), rec("A"))
        scheduler.add_task(make_task("B", dependencies=["A"]), rec("B"))
        scheduler.add_task(make_task("C", dependencies=["A"]), rec("C"))
        scheduler.add_task(
            make_task("D", dependencies=["B", "C"]), rec("D")
        )

        asyncio.run(scheduler.run())

        assert order[0] == "A"
        assert order[-1] == "D"


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    """Circular dependencies must raise CircularDependencyError."""

    def test_simple_cycle_detected(self):
        """t1 -> t2 -> t1"""
        scheduler = TaskScheduler()
        scheduler.add_task(
            make_task("t1", dependencies=["t2"]), always_succeed()
        )
        scheduler.add_task(
            make_task("t2", dependencies=["t1"]), always_succeed()
        )

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle_detected(self):
        """A -> B -> C -> A"""
        scheduler = TaskScheduler()
        scheduler.add_task(
            make_task("A", dependencies=["C"]), always_succeed()
        )
        scheduler.add_task(
            make_task("B", dependencies=["A"]), always_succeed()
        )
        scheduler.add_task(
            make_task("C", dependencies=["B"]), always_succeed()
        )

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_loop_cycle_detected(self):
        """t1 depends on itself."""
        scheduler = TaskScheduler()
        scheduler.add_task(
            make_task("t1", dependencies=["t1"]), always_succeed()
        )

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_no_false_positive_on_valid_dag(self):
        """A valid DAG must not raise."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("root"), always_succeed())
        scheduler.add_task(
            make_task("child", dependencies=["root"]), always_succeed()
        )

        plan = scheduler.get_execution_plan()   # Must not raise
        assert len(plan) == 2


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Failed tasks are retried up to max_retries with exponential back-off."""

    def test_task_retried_on_failure(self):
        """A coroutine that fails twice then succeeds is marked COMPLETED."""
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "recovered"

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, flaky)

        # Patch sleep so the test runs instantly
        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "recovered"
        assert t.retry_count == 2
        assert call_count == 3

    def test_task_fails_after_max_retries(self):
        """A task that always fails is marked FAILED after max_retries."""
        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=2)
        scheduler.add_task(t, always_fail())

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                asyncio.run(scheduler.run())

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 2

    def test_on_task_fail_event_emitted(self):
        """The on_task_fail event must fire when a task exhausts retries."""
        failed_ids: list[str] = []

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=1)
        scheduler.add_task(t, always_fail())
        scheduler.on("on_task_fail", lambda task: failed_ids.append(task.id))

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                asyncio.run(scheduler.run())

        assert "t1" in failed_ids

    def test_exponential_backoff_delays(self):
        """Sleep is called with exponentially increasing delays (2^attempt)."""
        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_fail_coro():
            raise RuntimeError("always fails")

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, always_fail_coro)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                asyncio.run(scheduler.run())

        # Back-off sequence: 2^0=1, 2^1=2, 2^2=4  (3 retries => 3 sleeps)
        assert sleep_calls == [1, 2, 4]

    def test_no_retry_on_success(self):
        """A task that succeeds on the first attempt should not sleep at all."""
        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), always_succeed())

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(scheduler.run())

        assert sleep_calls == []


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    """Scheduler must not exceed max_concurrency parallel tasks."""

    def test_concurrency_limit_respected(self):
        """
        With max_concurrency=2 and 4 slow tasks, no more than 2 should run
        simultaneously.
        """
        max_concurrent = 0
        currently_running = 0
        lock = asyncio.Lock()

        async def slow_task():
            nonlocal currently_running, max_concurrent
            async with lock:
                currently_running += 1
                if currently_running > max_concurrent:
                    max_concurrent = currently_running
            await asyncio.sleep(0.05)
            async with lock:
                currently_running -= 1
            return "done"

        scheduler = TaskScheduler(max_concurrency=2)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), slow_task)

        asyncio.run(scheduler.run())

        assert max_concurrent <= 2, (
            f"Expected max 2 concurrent tasks, observed {max_concurrent}"
        )

    def test_all_tasks_complete_despite_limit(self):
        """All tasks must still finish even with a tight concurrency cap."""
        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), always_succeed(i))

        results = asyncio.run(scheduler.run())

        assert len(results) == 6
        for i in range(6):
            assert results[f"t{i}"] == i

    def test_concurrency_limit_one_serialises_execution(self):
        """With max_concurrency=1 tasks run fully sequentially."""
        order: list[str] = []
        started: list[str] = []

        def make_coro(tid: str):
            async def _inner():
                started.append(tid)
                await asyncio.sleep(0.01)
                order.append(tid)
                return tid
            return _inner

        # Two independent tasks – with limit=1 they must not overlap
        scheduler = TaskScheduler(max_concurrency=1)
        scheduler.add_task(make_task("A", priority=10), make_coro("A"))
        scheduler.add_task(make_task("B", priority=1), make_coro("B"))

        asyncio.run(scheduler.run())

        # Verify they did not run concurrently: no task should have started
        # before the previous one finished (start order == finish order).
        assert started == order

    def test_invalid_concurrency_raises(self):
        """max_concurrency < 1 must be rejected at construction time."""
        with pytest.raises(ValueError):
            TaskScheduler(max_concurrency=0)
