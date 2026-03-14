"""
test_task_scheduler.py - pytest unit tests for task_scheduler.py
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, patch

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
    *,
    name: str = "",
    priority: int = 5,
    dependencies: list | None = None,
    max_retries: int = 3,
    fn=None,
) -> Task:
    """Factory for creating Task objects with an inline async callable."""

    async def _default():
        return f"result:{task_id}"

    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
        _fn=fn or _default,
    )


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    def test_single_task_completes(self):
        """A single task with no dependencies should reach COMPLETED status."""
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task)

        asyncio.run(scheduler.run())

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result:t1"

    def test_result_stored_on_task(self):
        """The return value of the callable must be stored in task.result."""
        scheduler = TaskScheduler()

        async def compute():
            return 42

        task = make_task("t1", fn=compute)
        scheduler.add_task(task)
        asyncio.run(scheduler.run())

        assert task.result == 42

    def test_multiple_independent_tasks_all_complete(self):
        """All tasks with no dependencies should complete."""
        scheduler = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(5)]
        for t in tasks:
            scheduler.add_task(t)

        asyncio.run(scheduler.run())

        for t in tasks:
            assert t.status == TaskStatus.COMPLETED

    def test_metrics_populated(self):
        """SchedulerMetrics should record per-task timing after a run."""
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task)
        metrics = asyncio.run(scheduler.run())

        assert "t1" in metrics.tasks
        assert metrics.tasks["t1"].elapsed >= 0
        assert metrics.total_elapsed >= 0


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_dependent_task_runs_after_dependency(self):
        """t2 depends on t1; t1 must finish before t2 starts."""
        order: List[str] = []
        scheduler = TaskScheduler()

        async def fn1():
            order.append("t1")
            return "t1"

        async def fn2():
            order.append("t2")
            return "t2"

        t1 = make_task("t1", fn=fn1)
        t2 = make_task("t2", dependencies=["t1"], fn=fn2)
        scheduler.add_task(t1)
        scheduler.add_task(t2)

        asyncio.run(scheduler.run())

        assert order.index("t1") < order.index("t2")

    def test_execution_plan_returns_correct_groups(self):
        """get_execution_plan should group tasks by dependency level."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b"))
        scheduler.add_task(make_task("c", dependencies=["a", "b"]))
        scheduler.add_task(make_task("d", dependencies=["c"]))

        plan = scheduler.get_execution_plan()

        # a and b can run first (order within group may vary)
        assert set(plan[0]) == {"a", "b"}
        assert plan[1] == ["c"]
        assert plan[2] == ["d"]

    def test_chain_dependency_three_tasks(self):
        """a -> b -> c should execute in strict serial order."""
        order: List[str] = []
        scheduler = TaskScheduler()

        for tid, deps in [("a", []), ("b", ["a"]), ("c", ["b"])]:
            async def fn(t=tid):
                order.append(t)
            scheduler.add_task(make_task(tid, dependencies=deps, fn=fn))

        asyncio.run(scheduler.run())

        assert order == ["a", "b", "c"]

    def test_failed_dependency_skips_downstream(self):
        """If t1 fails, t2 (which depends on t1) should not run."""
        scheduler = TaskScheduler()

        async def failing():
            raise RuntimeError("boom")

        t1 = make_task("t1", fn=failing, max_retries=0)
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1)
        scheduler.add_task(t2)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert t1.status == TaskStatus.FAILED
        # t2 must not have been executed; it stays PENDING
        assert t2.status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    def test_simple_cycle_raises(self):
        """a -> b -> a should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle_raises(self):
        """a -> b -> c -> a should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["b"]))

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_cycle_raises(self):
        """A task that depends on itself should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]))

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_no_cycle_does_not_raise(self):
        """A valid DAG must not raise any exception."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["a"]))
        scheduler.add_task(make_task("d", dependencies=["b", "c"]))

        # Should not raise
        plan = scheduler.get_execution_plan()
        assert len(plan) == 3


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_task_retried_on_failure(self):
        """A task that fails twice then succeeds should end as COMPLETED."""
        attempt = {"count": 0}

        async def flaky():
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise ValueError("not yet")
            return "ok"

        scheduler = TaskScheduler()
        task = make_task("t1", fn=flaky, max_retries=3)
        scheduler.add_task(task)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2
        assert task.result == "ok"

    def test_task_marked_failed_after_max_retries(self):
        """A task that always fails should be FAILED after max_retries attempts."""

        async def always_fail():
            raise RuntimeError("always")

        scheduler = TaskScheduler()
        task = make_task("t1", fn=always_fail, max_retries=2)
        scheduler.add_task(task)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert task.status == TaskStatus.FAILED
        # retry_count == max_retries + 1  (initial attempt + all retries)
        assert task.retry_count == task.max_retries + 1

    def test_on_task_fail_event_emitted(self):
        """on_task_fail observer should be called when a task exhausts retries."""
        failed_tasks = []

        async def always_fail():
            raise RuntimeError("bad")

        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, e: failed_tasks.append(t.id))

        task = make_task("t1", fn=always_fail, max_retries=0)
        scheduler.add_task(task)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert "t1" in failed_tasks

    def test_retry_count_tracked_in_metrics(self):
        """TaskMetrics.retry_count should reflect the number of retries."""
        attempt = {"n": 0}

        async def flaky():
            attempt["n"] += 1
            if attempt["n"] < 2:
                raise ValueError("retry me")
            return "done"

        scheduler = TaskScheduler()
        task = make_task("t1", fn=flaky, max_retries=3)
        scheduler.add_task(task)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            metrics = asyncio.run(scheduler.run())

        assert metrics.tasks["t1"].retry_count == 1

    def test_backoff_sleep_called_with_correct_delay(self):
        """asyncio.sleep should be called with exponential backoff values."""
        sleep_calls: List[float] = []

        async def always_fail():
            raise RuntimeError("always")

        scheduler = TaskScheduler()
        task = make_task("t1", fn=always_fail, max_retries=3)
        scheduler.add_task(task)

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(scheduler.run())

        # Backoffs for retries 1,2,3 should be 1,2,4
        assert sleep_calls == [1, 2, 4]


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrency_limit_respected(self):
        """No more than max_concurrency tasks should run at the same time."""
        max_concurrency = 2
        concurrent_peak = {"value": 0}
        active = {"count": 0}

        async def slow_task():
            active["count"] += 1
            concurrent_peak["value"] = max(concurrent_peak["value"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return "done"

        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}", fn=slow_task))

        asyncio.run(scheduler.run())

        assert concurrent_peak["value"] <= max_concurrency

    def test_independent_tasks_run_concurrently(self):
        """Independent tasks should overlap in time when concurrency > 1."""
        start_times: List[float] = []

        async def timed_task():
            start_times.append(time.monotonic())
            await asyncio.sleep(0.05)

        scheduler = TaskScheduler(max_concurrency=4)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}", fn=timed_task))

        asyncio.run(scheduler.run())

        # All four should start within a short window (< 0.02 s) of each other
        spread = max(start_times) - min(start_times)
        assert spread < 0.02, f"Tasks did not start concurrently; spread={spread:.3f}s"

    def test_observer_events_fire_for_every_task(self):
        """on_task_start and on_task_complete must fire once per successful task."""
        started = []
        completed = []

        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.on_task_complete(lambda t: completed.append(t.id))

        for i in range(3):
            scheduler.add_task(make_task(f"t{i}"))

        asyncio.run(scheduler.run())

        assert sorted(started) == ["t0", "t1", "t2"]
        assert sorted(completed) == ["t0", "t1", "t2"]
