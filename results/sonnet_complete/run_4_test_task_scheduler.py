"""
test_task_scheduler.py - pytest test suite for task_scheduler.py
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import MagicMock

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


async def success_coro(task: Task):
    """Instantly succeeds and returns the task id."""
    return task.id


async def slow_coro(task: Task):
    """Simulates work by sleeping for 0.05 s."""
    await asyncio.sleep(0.05)
    return task.id


# ---------------------------------------------------------------------------
# Test 1 – Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    def test_single_task_completes(self):
        scheduler = TaskScheduler()
        t = make_task("t1", name="basic")
        scheduler.add_task(t, success_coro)

        asyncio.run(scheduler.run())

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "t1"

    def test_multiple_independent_tasks_all_complete(self):
        scheduler = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(4)]
        for t in tasks:
            scheduler.add_task(t, success_coro)

        asyncio.run(scheduler.run())

        assert all(t.status == TaskStatus.COMPLETED for t in tasks)

    def test_metrics_total_time_is_populated(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), success_coro)
        metrics = asyncio.run(scheduler.run())
        assert metrics.total_time is not None
        assert metrics.total_time >= 0

    def test_per_task_metrics_populated(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), success_coro)
        metrics = asyncio.run(scheduler.run())
        tm = metrics.per_task["t1"]
        assert tm.elapsed is not None
        assert tm.elapsed >= 0


# ---------------------------------------------------------------------------
# Test 2 – Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_dependent_task_runs_after_dependency(self):
        execution_order: List[str] = []

        async def recording_coro(task: Task):
            execution_order.append(task.id)
            return task.id

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1, recording_coro)
        scheduler.add_task(t2, recording_coro)

        asyncio.run(scheduler.run())

        assert execution_order.index("t1") < execution_order.index("t2")

    def test_execution_plan_groups_are_ordered(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), success_coro)
        scheduler.add_task(make_task("b", dependencies=["a"]), success_coro)
        scheduler.add_task(make_task("c", dependencies=["b"]), success_coro)

        plan = scheduler.get_execution_plan()

        # Flatten and verify order
        flat = [tid for group in plan for tid in group]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    def test_independent_tasks_in_same_group(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("x"), success_coro)
        scheduler.add_task(make_task("y"), success_coro)
        scheduler.add_task(make_task("z"), success_coro)

        plan = scheduler.get_execution_plan()

        # All three have no dependencies → they should be in the first group
        first_group = set(plan[0])
        assert {"x", "y", "z"} == first_group

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["ghost"]), success_coro)

        with pytest.raises(TaskNotFoundError):
            asyncio.run(scheduler.run())


# ---------------------------------------------------------------------------
# Test 3 – Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_direct_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), success_coro)
        scheduler.add_task(make_task("b", dependencies=["a"]), success_coro)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_indirect_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]), success_coro)
        scheduler.add_task(make_task("b", dependencies=["a"]), success_coro)
        scheduler.add_task(make_task("c", dependencies=["b"]), success_coro)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]), success_coro)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_no_cycle_does_not_raise(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), success_coro)
        scheduler.add_task(make_task("b", dependencies=["a"]), success_coro)
        scheduler.add_task(make_task("c", dependencies=["a"]), success_coro)
        scheduler.add_task(make_task("d", dependencies=["b", "c"]), success_coro)

        # Must not raise
        plan = scheduler.get_execution_plan()
        assert len(plan) > 0


# ---------------------------------------------------------------------------
# Test 4 – Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_task_retries_up_to_max_and_fails(self):
        call_counts: dict[str, int] = {"n": 0}

        async def always_fails(task: Task):
            call_counts["n"] += 1
            raise RuntimeError("boom")

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=2)
        scheduler.add_task(t, always_fails)

        # Patch sleep so the test runs fast
        async def instant_sleep(_):
            pass

        original_sleep = asyncio.sleep

        async def run_with_patched_sleep():
            asyncio.sleep = instant_sleep
            try:
                await scheduler.run()
            finally:
                asyncio.sleep = original_sleep

        asyncio.run(run_with_patched_sleep())

        assert t.status == TaskStatus.FAILED
        # Initial attempt + 2 retries = 3 total calls
        assert call_counts["n"] == 3
        assert t.retry_count == 3

    def test_task_succeeds_on_second_attempt(self):
        call_counts: dict[str, int] = {"n": 0}

        async def fails_once(task: Task):
            call_counts["n"] += 1
            if call_counts["n"] == 1:
                raise RuntimeError("first failure")
            return "ok"

        async def instant_sleep(_):
            pass

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, fails_once)

        original_sleep = asyncio.sleep

        async def run_with_patched_sleep():
            asyncio.sleep = instant_sleep
            try:
                await scheduler.run()
            finally:
                asyncio.sleep = original_sleep

        asyncio.run(run_with_patched_sleep())

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "ok"
        assert t.retry_count == 1

    def test_retry_count_tracked_in_metrics(self):
        async def fails_twice(task: Task):
            if task.retry_count < 2:
                raise RuntimeError("not yet")
            return "done"

        async def instant_sleep(_):
            pass

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, fails_twice)

        original_sleep = asyncio.sleep

        async def run_with_patched_sleep():
            asyncio.sleep = instant_sleep
            try:
                await scheduler.run()
            finally:
                asyncio.sleep = original_sleep

        asyncio.run(run_with_patched_sleep())

        assert t.status == TaskStatus.COMPLETED
        assert scheduler.metrics.per_task["t1"].retry_count == 2


# ---------------------------------------------------------------------------
# Test 5 – Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    def test_concurrency_limit_respected(self):
        """Verify that at most `max_concurrency` tasks run simultaneously."""
        active: list[int] = [0]
        peak: list[int] = [0]

        async def counting_coro(task: Task):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.05)
            active[0] -= 1
            return task.id

        max_concurrency = 2
        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), counting_coro)

        asyncio.run(scheduler.run())

        assert peak[0] <= max_concurrency

    def test_higher_concurrency_finishes_faster(self):
        """More concurrency → faster wall-clock time for independent tasks."""

        async def slow(task: Task):
            await asyncio.sleep(0.05)
            return task.id

        def run_with_concurrency(limit: int) -> float:
            scheduler = TaskScheduler(max_concurrency=limit)
            for i in range(4):
                scheduler.add_task(make_task(f"t{i}"), slow)
            start = time.monotonic()
            asyncio.run(scheduler.run())
            return time.monotonic() - start

        serial_time = run_with_concurrency(1)
        parallel_time = run_with_concurrency(4)

        # Parallel should be at least 2× faster
        assert parallel_time < serial_time / 2

    def test_observer_events_fired(self):
        started: List[str] = []
        completed: List[str] = []
        failed: List[str] = []

        async def instant_sleep(_):
            pass

        scheduler = TaskScheduler()

        async def bad_coro(task: Task):
            raise RuntimeError("oops")

        t_good = make_task("good")
        t_bad = make_task("bad", max_retries=0)
        scheduler.add_task(t_good, success_coro)
        scheduler.add_task(t_bad, bad_coro)

        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.on_task_complete(lambda t: completed.append(t.id))
        scheduler.on_task_fail(lambda t: failed.append(t.id))

        original_sleep = asyncio.sleep

        async def run():
            asyncio.sleep = instant_sleep
            try:
                await scheduler.run()
            finally:
                asyncio.sleep = original_sleep

        asyncio.run(run())

        assert "good" in started
        assert "bad" in started
        assert "good" in completed
        assert "bad" in failed
        assert "good" not in failed
        assert "bad" not in completed
