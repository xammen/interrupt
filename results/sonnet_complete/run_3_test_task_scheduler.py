"""
test_task_scheduler.py - pytest test suite for task_scheduler.py
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


async def ok_handler(task: Task):
    """Handler that always succeeds immediately."""
    return f"result-{task.id}"


# ---------------------------------------------------------------------------
# Test 1 – Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, ok_handler)

        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "result-t1"
        assert metrics.total_time is not None
        assert metrics.total_time >= 0

    @pytest.mark.asyncio
    async def test_result_is_stored_on_task(self):
        scheduler = TaskScheduler()
        t = make_task("t2")

        async def handler(task):
            return {"value": 42}

        scheduler.add_task(t, handler)
        await scheduler.run()

        assert t.result == {"value": 42}

    @pytest.mark.asyncio
    async def test_per_task_metrics_recorded(self):
        scheduler = TaskScheduler()
        t = make_task("t3")
        scheduler.add_task(t, ok_handler)

        metrics = await scheduler.run()

        assert "t3" in metrics.per_task
        task_m = metrics.per_task["t3"]
        assert task_m.elapsed is not None
        assert task_m.elapsed >= 0


# ---------------------------------------------------------------------------
# Test 2 – Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_runs_after_dependency(self):
        execution_order: List[str] = []
        scheduler = TaskScheduler()

        t_a = make_task("A")
        t_b = make_task("B", dependencies=["A"])

        async def track(task):
            execution_order.append(task.id)
            return task.id

        scheduler.add_task(t_a, track)
        scheduler.add_task(t_b, track)
        await scheduler.run()

        assert execution_order.index("A") < execution_order.index("B")

    @pytest.mark.asyncio
    async def test_chain_a_b_c(self):
        order: List[str] = []
        scheduler = TaskScheduler()

        tasks = [
            make_task("A"),
            make_task("B", dependencies=["A"]),
            make_task("C", dependencies=["B"]),
        ]

        async def track(task):
            order.append(task.id)

        for t in tasks:
            scheduler.add_task(t, track)

        await scheduler.run()
        assert order == ["A", "B", "C"]

    def test_execution_plan_groups(self):
        """get_execution_plan groups independent tasks in the same layer."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("root"), ok_handler)
        scheduler.add_task(make_task("child1", dependencies=["root"]), ok_handler)
        scheduler.add_task(make_task("child2", dependencies=["root"]), ok_handler)

        plan = scheduler.get_execution_plan()

        assert plan[0] == ["root"]
        assert set(plan[1]) == {"child1", "child2"}

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_diamond_dependency(self):
        """A → B, A → C, B → D, C → D (diamond shape)."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("A"), ok_handler)
        scheduler.add_task(make_task("B", dependencies=["A"]), ok_handler)
        scheduler.add_task(make_task("C", dependencies=["A"]), ok_handler)
        scheduler.add_task(make_task("D", dependencies=["B", "C"]), ok_handler)

        await scheduler.run()

        for tid in ("A", "B", "C", "D"):
            assert scheduler._tasks[tid].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test 3 – Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    def test_direct_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("X", dependencies=["Y"]), ok_handler)
        scheduler.add_task(make_task("Y", dependencies=["X"]), ok_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_indirect_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("A", dependencies=["C"]), ok_handler)
        scheduler.add_task(make_task("B", dependencies=["A"]), ok_handler)
        scheduler.add_task(make_task("C", dependencies=["B"]), ok_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_loop_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("self_loop", dependencies=["self_loop"]), ok_handler)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("p", dependencies=["q"]), ok_handler)
        scheduler.add_task(make_task("q", dependencies=["p"]), ok_handler)

        with pytest.raises(CircularDependencyError):
            await scheduler.run()


# ---------------------------------------------------------------------------
# Test 4 – Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_then_succeeds(self):
        """Fail twice, succeed on the third attempt."""
        call_count = 0

        async def flaky_handler(task: Task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "ok"

        scheduler = TaskScheduler(base_backoff=0.0)  # zero wait for speed
        t = make_task("flaky", max_retries=3)
        scheduler.add_task(t, flaky_handler)

        await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.retry_count == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        async def always_fail(task: Task):
            raise RuntimeError("always bad")

        scheduler = TaskScheduler(base_backoff=0.0)
        t = make_task("always_fail", max_retries=2)
        scheduler.add_task(t, always_fail)

        with pytest.raises(RuntimeError):
            await scheduler.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self):
        """Verify that asyncio.sleep is called with doubling delays."""
        sleep_calls: List[float] = []

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_fail(task: Task):
            raise RuntimeError("fail")

        scheduler = TaskScheduler(base_backoff=1.0)
        t = make_task("backoff_task", max_retries=3)
        scheduler.add_task(t, always_fail)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await scheduler.run()

        # Expected delays: 1s, 2s, 4s  (base * 2^(attempt-1))
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_retry_count_tracked_in_metrics(self):
        call_count = 0

        async def fail_once(task: Task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("once")
            return "ok"

        scheduler = TaskScheduler(base_backoff=0.0)
        t = make_task("metric_retry", max_retries=2)
        scheduler.add_task(t, fail_once)

        metrics = await scheduler.run()

        assert metrics.per_task["metric_retry"].retry_count == 1


# ---------------------------------------------------------------------------
# Test 5 – Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """Never more than N tasks run simultaneously."""
        max_concurrency = 2
        concurrent_peak = 0
        currently_running = 0

        async def slow_handler(task: Task):
            nonlocal concurrent_peak, currently_running
            currently_running += 1
            concurrent_peak = max(concurrent_peak, currently_running)
            await asyncio.sleep(0.05)
            currently_running -= 1

        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), slow_handler)

        await scheduler.run()

        assert concurrent_peak <= max_concurrency

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """With no dependencies, tasks should overlap (finish faster than serial)."""
        scheduler = TaskScheduler(max_concurrency=4)

        async def slow_handler(task: Task):
            await asyncio.sleep(0.1)

        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), slow_handler)

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # Serial would take ~0.4 s; concurrent should be ~0.1 s (+overhead)
        assert elapsed < 0.35, f"Expected concurrent execution, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_concurrency_limit_one_runs_serially(self):
        """max_concurrency=1 forces fully serial execution."""
        order: List[float] = []

        async def timed_handler(task: Task):
            order.append(time.monotonic())
            await asyncio.sleep(0.05)

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(3):
            scheduler.add_task(make_task(f"s{i}"), timed_handler)

        await scheduler.run()

        # Each task must start after the previous finishes (~50 ms gap)
        for i in range(1, len(order)):
            gap = order[i] - order[i - 1]
            assert gap >= 0.04, f"Gap {gap:.3f}s too small; tasks overlapped"


# ---------------------------------------------------------------------------
# Test 6 – Observer / event callbacks
# ---------------------------------------------------------------------------

class TestObserverPattern:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started: List[str] = []

        async def on_start(task: Task):
            started.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_start(on_start)
        scheduler.add_task(make_task("ev1"), ok_handler)

        await scheduler.run()
        assert "ev1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        completed: List[str] = []

        async def on_complete(task: Task):
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_complete(on_complete)
        scheduler.add_task(make_task("ev2"), ok_handler)

        await scheduler.run()
        assert "ev2" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called(self):
        failed: List[str] = []

        async def on_fail(task: Task):
            failed.append(task.id)

        async def bad_handler(task: Task):
            raise RuntimeError("boom")

        scheduler = TaskScheduler(base_backoff=0.0)
        scheduler.on_task_fail(on_fail)
        scheduler.add_task(make_task("ev3", max_retries=0), bad_handler)

        with pytest.raises(RuntimeError):
            await scheduler.run()

        assert "ev3" in failed
