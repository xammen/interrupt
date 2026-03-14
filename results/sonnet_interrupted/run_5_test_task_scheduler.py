"""
test_task_scheduler.py - pytest test suite for task_scheduler.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List
from unittest.mock import AsyncMock

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
    id: str,
    name: str = "",
    priority: int = 5,
    dependencies: list | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=id,
        name=name or id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def success_handler(task: Task) -> str:
    """Always succeeds immediately."""
    return f"result_{task.id}"


def make_flaky_handler(fail_times: int):
    """Returns a handler that raises on the first *fail_times* calls."""
    calls = {"n": 0}

    async def handler(task: Task) -> str:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(f"Simulated failure #{calls['n']}")
        return f"result_{task.id}"

    return handler


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        """A single task with no dependencies should reach COMPLETED status."""
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.register(task, success_handler)

        await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result_t1"

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        """All tasks with no dependencies should complete."""
        scheduler = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(5)]
        for t in tasks:
            scheduler.register(t, success_handler)

        await scheduler.run()

        assert all(t.status == TaskStatus.COMPLETED for t in tasks)

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self):
        """The return value of the handler should be stored in task.result."""
        scheduler = TaskScheduler()
        task = make_task("t1")

        async def handler(t: Task) -> dict:
            return {"answer": 42}

        scheduler.register(task, handler)
        await scheduler.run()

        assert task.result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_metrics_recorded(self):
        """Elapsed time should be recorded for each task."""
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.register(task, success_handler)
        await scheduler.run()

        m = scheduler.metrics.per_task["t1"]
        assert m.elapsed is not None
        assert m.elapsed >= 0
        assert scheduler.metrics.total_time is not None


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        """t2 depends on t1; t1 must complete before t2 starts."""
        order: List[str] = []

        async def tracking_handler(task: Task) -> str:
            order.append(task.id)
            return task.id

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.register(t1, tracking_handler)
        scheduler.register(t2, tracking_handler)

        await scheduler.run()

        assert order.index("t1") < order.index("t2")

    @pytest.mark.asyncio
    async def test_execution_plan_groups(self):
        """get_execution_plan should return correct dependency groups."""
        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2")
        t3 = make_task("t3", dependencies=["t1", "t2"])
        for t in (t1, t2, t3):
            scheduler.register(t, success_handler)

        plan = scheduler.get_execution_plan()

        # First group: t1, t2 (no deps); second group: t3
        first_ids = set(plan[0])
        assert "t1" in first_ids
        assert "t2" in first_ids
        assert plan[1] == ["t3"]

    @pytest.mark.asyncio
    async def test_chain_dependency(self):
        """t1 -> t2 -> t3: execution order must be strictly sequential."""
        order: List[str] = []

        async def tracking_handler(task: Task) -> str:
            order.append(task.id)
            return task.id

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t2"])
        for t in (t1, t2, t3):
            scheduler.register(t, tracking_handler)

        await scheduler.run()

        assert order == ["t1", "t2", "t3"]

    def test_unknown_dependency_raises_value_error(self):
        """Referencing a non-existent task ID in dependencies should raise."""
        scheduler = TaskScheduler()
        task = make_task("t1", dependencies=["nonexistent"])
        scheduler.register(task, success_handler)

        with pytest.raises(ValueError, match="unknown task"):
            scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_direct_cycle(self):
        """t1 -> t2 -> t1 should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t2"])
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.register(t1, success_handler)
        scheduler.register(t2, success_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_indirect_cycle(self):
        """t1 -> t2 -> t3 -> t1 should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t3"])
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t2"])
        for t in (t1, t2, t3):
            scheduler.register(t, success_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        """Calling run() on a scheduler with a cycle should also raise."""
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t2"])
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.register(t1, success_handler)
        scheduler.register(t2, success_handler)

        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_self_dependency_raises(self):
        """A task depending on itself is a trivial cycle."""
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t1"])
        scheduler.register(t1, success_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_then_succeeds(self):
        """A task that fails twice and then succeeds should be COMPLETED."""
        scheduler = TaskScheduler(base_retry_delay=0.0)
        task = make_task("t1", max_retries=3)
        scheduler.register(task, make_flaky_handler(fail_times=2))

        await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        """A task that always fails should end up FAILED after max_retries."""
        scheduler = TaskScheduler(base_retry_delay=0.0)
        task = make_task("t1", max_retries=2)
        scheduler.register(task, make_flaky_handler(fail_times=999))

        await scheduler.run()

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == task.max_retries + 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Delays should roughly follow base * 2^attempt pattern."""
        delays: List[float] = []
        base = 0.05

        scheduler = TaskScheduler(base_retry_delay=base)
        task = make_task("t1", max_retries=3)
        fail_count = {"n": 0}

        async def flaky(t: Task) -> str:
            fail_count["n"] += 1
            if fail_count["n"] <= 3:
                raise RuntimeError("fail")
            return "ok"

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)
            # Don't actually sleep to keep tests fast
            return

        import unittest.mock as mock

        with mock.patch("asyncio.sleep", side_effect=mock_sleep):
            scheduler.register(task, flaky)
            await scheduler.run()

        assert len(delays) == 3
        assert delays[0] == pytest.approx(base * 1, rel=0.01)
        assert delays[1] == pytest.approx(base * 2, rel=0.01)
        assert delays[2] == pytest.approx(base * 4, rel=0.01)

    @pytest.mark.asyncio
    async def test_on_task_fail_event_emitted(self):
        """on_task_fail callback must be invoked when a task is permanently failed."""
        scheduler = TaskScheduler(base_retry_delay=0.0)
        task = make_task("t1", max_retries=0)
        failed_tasks: List[str] = []

        async def fail_cb(t: Task) -> None:
            failed_tasks.append(t.id)

        scheduler.on_task_fail(fail_cb)
        scheduler.register(task, make_flaky_handler(fail_times=999))

        await scheduler.run()

        assert "t1" in failed_tasks


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_concurrency_limit_not_exceeded(self):
        """No more than concurrency_limit tasks should run simultaneously."""
        limit = 3
        max_concurrent = {"count": 0, "peak": 0}
        lock = asyncio.Lock()

        async def slow_handler(task: Task) -> str:
            async with lock:
                max_concurrent["count"] += 1
                max_concurrent["peak"] = max(
                    max_concurrent["peak"], max_concurrent["count"]
                )
            await asyncio.sleep(0.05)
            async with lock:
                max_concurrent["count"] -= 1
            return task.id

        scheduler = TaskScheduler(concurrency_limit=limit)
        for i in range(10):
            t = make_task(f"t{i}")
            scheduler.register(t, slow_handler)

        await scheduler.run()

        assert max_concurrent["peak"] <= limit

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Independent tasks should execute in parallel, not sequentially."""
        n = 4
        sleep_duration = 0.1

        scheduler = TaskScheduler(concurrency_limit=n)
        tasks = [make_task(f"t{i}") for i in range(n)]

        async def slow_handler(task: Task) -> str:
            await asyncio.sleep(sleep_duration)
            return task.id

        for t in tasks:
            scheduler.register(t, slow_handler)

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # If truly concurrent, total time << n * sleep_duration
        assert elapsed < sleep_duration * n * 0.75

    @pytest.mark.asyncio
    async def test_observer_events_fired_correctly(self):
        """on_task_start and on_task_complete must both fire for every task."""
        started: List[str] = []
        completed: List[str] = []

        async def start_cb(t: Task) -> None:
            started.append(t.id)

        async def complete_cb(t: Task) -> None:
            completed.append(t.id)

        scheduler = TaskScheduler()
        scheduler.on_task_start(start_cb)
        scheduler.on_task_complete(complete_cb)

        for i in range(3):
            t = make_task(f"t{i}")
            scheduler.register(t, success_handler)

        await scheduler.run()

        assert sorted(started) == ["t0", "t1", "t2"]
        assert sorted(completed) == ["t0", "t1", "t2"]
