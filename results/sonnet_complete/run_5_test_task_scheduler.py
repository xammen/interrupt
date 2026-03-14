"""
Pytest test suite for task_scheduler.py.

Run with:
    pytest test_task_scheduler.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock

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

def make_task(
    task_id: str,
    name: str = "task",
    priority: int = 5,
    dependencies: List[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def noop() -> str:
    """Trivial coroutine that succeeds immediately."""
    return "ok"


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    """A registered task should run, update its status, and store a result."""

    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        task = make_task("t1", name="hello")
        scheduler.add_task(task, noop)

        await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "ok"

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self):
        scheduler = TaskScheduler()
        task = make_task("t1")

        async def compute():
            return 42

        scheduler.add_task(task, compute)
        await scheduler.run()

        assert task.result == 42

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, noop)
        metrics = await scheduler.run()

        m = metrics["t1"]
        assert m.start_time > 0
        assert m.end_time >= m.start_time
        assert m.elapsed >= 0

    @pytest.mark.asyncio
    async def test_observer_on_task_start_called(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        started: List[str] = []

        def on_start(t: Task) -> None:
            started.append(t.id)

        scheduler.on("on_task_start", on_start)
        scheduler.add_task(task, noop)
        await scheduler.run()

        assert "t1" in started

    @pytest.mark.asyncio
    async def test_observer_on_task_complete_called(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        completed: List[str] = []

        def on_complete(t: Task) -> None:
            completed.append(t.id)

        scheduler.on("on_task_complete", on_complete)
        scheduler.add_task(task, noop)
        await scheduler.run()

        assert "t1" in completed


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    """Tasks must wait for their dependencies to complete first."""

    @pytest.mark.asyncio
    async def test_dependent_runs_after_dependency(self):
        execution_order: List[str] = []

        async def record(task_id: str):
            async def fn():
                execution_order.append(task_id)
            return fn

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])

        scheduler.add_task(t1, await record("t1"))
        scheduler.add_task(t2, await record("t2"))
        await scheduler.run()

        assert execution_order.index("t1") < execution_order.index("t2")

    def test_get_execution_plan_returns_correct_groups(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t1"])
        t4 = make_task("t4", dependencies=["t2", "t3"])

        scheduler.add_task(t1, noop)
        scheduler.add_task(t2, noop)
        scheduler.add_task(t3, noop)
        scheduler.add_task(t4, noop)

        plan = scheduler.get_execution_plan()

        # Group 0 must contain only t1
        assert plan[0] == ["t1"]
        # Group 1 must contain t2 and t3 (any order)
        assert set(plan[1]) == {"t2", "t3"}
        # Group 2 must contain only t4
        assert plan[2] == ["t4"]

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_chain(self):
        scheduler = TaskScheduler()
        ids = ["a", "b", "c", "d"]
        for i, tid in enumerate(ids):
            deps = [ids[i - 1]] if i > 0 else []
            scheduler.add_task(make_task(tid, dependencies=deps), noop)

        await scheduler.run()

        for tid in ids:
            assert scheduler._tasks[tid].status == TaskStatus.COMPLETED

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1", dependencies=["ghost"]), noop)
        with pytest.raises(ValueError, match="unknown task"):
            scheduler.get_execution_plan()

    def test_priority_ordering_within_group(self):
        """Higher-priority tasks should appear first within the same group."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("low", priority=2), noop)
        scheduler.add_task(make_task("high", priority=9), noop)

        plan = scheduler.get_execution_plan()
        group = plan[0]
        assert group.index("high") < group.index("low")


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    """Cycles in the dependency graph must be detected and reported."""

    def test_direct_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), noop)
        scheduler.add_task(make_task("b", dependencies=["a"]), noop)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_indirect_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("x", dependencies=["z"]), noop)
        scheduler.add_task(make_task("y", dependencies=["x"]), noop)
        scheduler.add_task(make_task("z", dependencies=["y"]), noop)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_cycle_error_contains_node_names(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("p", dependencies=["q"]), noop)
        scheduler.add_task(make_task("q", dependencies=["p"]), noop)

        try:
            scheduler.get_execution_plan()
            pytest.fail("Expected CircularDependencyError")
        except CircularDependencyError as exc:
            assert "p" in exc.cycle or "q" in exc.cycle

    def test_self_cycle_raises(self):
        scheduler = TaskScheduler()
        # A task that depends on itself
        scheduler.add_task(make_task("self_ref", dependencies=["self_ref"]), noop)

        with pytest.raises((CircularDependencyError, ValueError)):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_cycle_detected_at_run_time(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), noop)
        scheduler.add_task(make_task("b", dependencies=["a"]), noop)

        with pytest.raises(CircularDependencyError):
            await scheduler.run()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential back-off
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Failed tasks must be retried with exponential back-off."""

    @pytest.mark.asyncio
    async def test_task_retried_on_failure(self):
        attempts = 0

        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient error")
            return "recovered"

        scheduler = TaskScheduler()
        task = make_task("t1", max_retries=3)
        scheduler.add_task(task, flaky)

        # Patch sleep to avoid real delays
        async def fast_sleep(_: float) -> None:
            pass

        asyncio.sleep  # reference before patch
        original_sleep = asyncio.sleep

        try:
            asyncio.sleep = fast_sleep  # type: ignore[assignment]
            await scheduler.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "recovered"
        assert task.retry_count == 2  # failed twice before success

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self):
        async def always_fail():
            raise ValueError("permanent error")

        scheduler = TaskScheduler()
        task = make_task("t1", max_retries=2)
        scheduler.add_task(task, always_fail)

        original_sleep = asyncio.sleep

        async def fast_sleep(_: float) -> None:
            pass

        try:
            asyncio.sleep = fast_sleep  # type: ignore[assignment]
            with pytest.raises(ValueError, match="permanent error"):
                await scheduler.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2

    @pytest.mark.asyncio
    async def test_on_task_fail_observer_called(self):
        failed_ids: List[str] = []

        def on_fail(t: Task) -> None:
            failed_ids.append(t.id)

        async def bad():
            raise RuntimeError("boom")

        scheduler = TaskScheduler()
        task = make_task("bad_task", max_retries=0)
        scheduler.on("on_task_fail", on_fail)
        scheduler.add_task(task, bad)

        with pytest.raises(RuntimeError):
            await scheduler.run()

        assert "bad_task" in failed_ids

    @pytest.mark.asyncio
    async def test_backoff_sleep_called_with_increasing_delays(self):
        """Verify sleep durations follow the 2**attempt pattern."""
        sleep_calls: List[float] = []

        async def recording_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        scheduler = TaskScheduler()
        task = make_task("t1", max_retries=3)
        scheduler.add_task(task, flaky)

        original_sleep = asyncio.sleep
        try:
            asyncio.sleep = recording_sleep  # type: ignore[assignment]
            with pytest.raises(RuntimeError):
                await scheduler.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        # Expect sleeps: 1, 2, 4 (for attempts 0, 1, 2 before final failure)
        assert sleep_calls == [1, 2, 4]

    @pytest.mark.asyncio
    async def test_retry_count_tracked_in_metrics(self):
        async def always_fail():
            raise RuntimeError("x")

        scheduler = TaskScheduler()
        task = make_task("t1", max_retries=2)
        scheduler.add_task(task, always_fail)

        original_sleep = asyncio.sleep

        async def fast_sleep(_: float) -> None:
            pass

        try:
            asyncio.sleep = fast_sleep  # type: ignore[assignment]
            with pytest.raises(RuntimeError):
                await scheduler.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        metrics = scheduler.get_metrics()
        assert metrics["t1"].retries == 2


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    """No more than max_concurrency tasks should run simultaneously."""

    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """Track the high-water mark of simultaneous running tasks."""
        concurrency_limit = 2
        running_now = 0
        high_water = 0
        lock = asyncio.Lock()

        async def measured():
            nonlocal running_now, high_water
            async with lock:
                running_now += 1
                if running_now > high_water:
                    high_water = running_now
            await asyncio.sleep(0.05)  # hold the slot briefly
            async with lock:
                running_now -= 1

        scheduler = TaskScheduler(max_concurrency=concurrency_limit)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), measured)

        await scheduler.run()

        assert high_water <= concurrency_limit

    @pytest.mark.asyncio
    async def test_all_tasks_complete_under_concurrency_limit(self):
        scheduler = TaskScheduler(max_concurrency=2)
        task_ids = [f"t{i}" for i in range(5)]
        for tid in task_ids:
            scheduler.add_task(make_task(tid), noop)

        await scheduler.run()

        for tid in task_ids:
            assert scheduler._tasks[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrency_limit_one_serialises_execution(self):
        """With max_concurrency=1 no two tasks should overlap."""
        running_simultaneously = 0
        overlap_detected = False
        lock = asyncio.Lock()

        async def measured():
            nonlocal running_simultaneously, overlap_detected
            async with lock:
                running_simultaneously += 1
                if running_simultaneously > 1:
                    overlap_detected = True
            await asyncio.sleep(0.02)
            async with lock:
                running_simultaneously -= 1

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), measured)

        await scheduler.run()

        assert not overlap_detected

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Without a concurrency limit, independent tasks should overlap in time."""
        start_times: List[float] = []
        lock = asyncio.Lock()

        async def timed():
            async with lock:
                start_times.append(time.monotonic())
            await asyncio.sleep(0.05)

        scheduler = TaskScheduler(max_concurrency=4)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), timed)

        await scheduler.run()

        # All 4 tasks should have started within a small window of each other
        spread = max(start_times) - min(start_times)
        assert spread < 0.04, f"Tasks did not start concurrently; spread={spread:.3f}s"

    def test_invalid_concurrency_limit_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            TaskScheduler(max_concurrency=0)
