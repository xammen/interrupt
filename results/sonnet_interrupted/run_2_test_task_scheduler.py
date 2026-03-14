"""
test_task_scheduler.py - pytest suite for task_scheduler.py

Run with:
    pytest test_task_scheduler.py -v
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
    tid: str,
    name: str = "",
    priority: int = 5,
    dependencies: List[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=tid,
        name=name or tid,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def success_coro(value: int = 42):
    """A trivial coroutine that succeeds immediately."""
    return value


def factory(value: int = 42):
    async def _coro():
        return value
    return _coro


def slow_factory(delay: float = 0.05):
    async def _coro():
        await asyncio.sleep(delay)
        return delay
    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    def test_single_task_completes(self):
        """A single task with no dependencies should reach COMPLETED status."""
        scheduler = TaskScheduler()
        task = make_task("t1", priority=5)
        scheduler.add_task(task, factory(99))

        results = asyncio.run(scheduler.run())

        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t1"].result == 99

    def test_multiple_independent_tasks_all_complete(self):
        """All independent tasks should complete successfully."""
        scheduler = TaskScheduler()
        for i in range(4):
            t = make_task(f"t{i}", priority=i + 1)
            scheduler.add_task(t, factory(i))

        results = asyncio.run(scheduler.run())

        for i in range(4):
            assert results[f"t{i}"].status == TaskStatus.COMPLETED
            assert results[f"t{i}"].result == i

    def test_metrics_populated_after_run(self):
        """Execution metrics should record start/end times for each task."""
        scheduler = TaskScheduler()
        task = make_task("m1")
        scheduler.add_task(task, factory(1))
        asyncio.run(scheduler.run())

        tm = scheduler.metrics.per_task["m1"]
        assert tm.start_time is not None
        assert tm.end_time is not None
        assert tm.elapsed is not None and tm.elapsed >= 0
        assert scheduler.metrics.total_elapsed is not None

    def test_observer_on_task_complete_called(self):
        """on_task_complete callback must be invoked when a task succeeds."""
        scheduler = TaskScheduler()
        task = make_task("obs1")
        scheduler.add_task(task, factory(7))

        fired: List[str] = []

        async def handler(t: Task):
            fired.append(t.id)

        scheduler.on_task_complete(handler)
        asyncio.run(scheduler.run())

        assert "obs1" in fired

    def test_priority_field_validation(self):
        """Task priority outside 1–10 must raise ValueError."""
        with pytest.raises(ValueError):
            make_task("bad", priority=0)
        with pytest.raises(ValueError):
            make_task("bad", priority=11)


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_dependent_task_runs_after_dependency(self):
        """A task must not start before its dependency has completed."""
        order: List[str] = []

        def tracking_factory(tid: str, delay: float = 0.0):
            async def _coro():
                await asyncio.sleep(delay)
                order.append(tid)
                return tid
            return _coro

        scheduler = TaskScheduler()
        t1 = make_task("base", priority=5)
        t2 = make_task("child", priority=5, dependencies=["base"])
        scheduler.add_task(t1, tracking_factory("base", delay=0.01))
        scheduler.add_task(t2, tracking_factory("child"))

        asyncio.run(scheduler.run())

        assert order.index("base") < order.index("child")

    def test_get_execution_plan_ordering(self):
        """get_execution_plan should return waves respecting dependencies."""
        scheduler = TaskScheduler()
        t1 = make_task("a", priority=5)
        t2 = make_task("b", priority=5, dependencies=["a"])
        t3 = make_task("c", priority=5, dependencies=["b"])
        for t in (t1, t2, t3):
            scheduler.add_task(t, factory())

        plan = scheduler.get_execution_plan()

        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_wave_priority_ordering(self):
        """Within a wave, higher-priority tasks should appear first."""
        scheduler = TaskScheduler()
        t_low = make_task("low", priority=2)
        t_high = make_task("high", priority=8)
        for t in (t_low, t_high):
            scheduler.add_task(t, factory())

        plan = scheduler.get_execution_plan()
        # Both tasks are independent → single wave
        assert plan[0][0] == "high"
        assert plan[0][1] == "low"

    def test_failed_dependency_cascades(self):
        """If a dependency fails, dependent tasks should be marked FAILED too."""
        fail_count = 0

        def always_fail():
            async def _coro():
                raise RuntimeError("forced failure")
            return _coro

        scheduler = TaskScheduler(base_backoff=0.0)
        t1 = make_task("root", max_retries=0)
        t2 = make_task("child", dependencies=["root"])
        scheduler.add_task(t1, always_fail())
        scheduler.add_task(t2, factory())

        async def on_fail(t: Task):
            nonlocal fail_count
            fail_count += 1

        scheduler.on_task_fail(on_fail)
        results = asyncio.run(scheduler.run())

        assert results["root"].status == TaskStatus.FAILED
        assert results["child"].status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_direct_cycle_raises(self):
        """A ↔ B should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        t_a = make_task("A", dependencies=["B"])
        t_b = make_task("B", dependencies=["A"])
        for t in (t_a, t_b):
            scheduler.add_task(t, factory())

        with pytest.raises(CircularDependencyError):
            asyncio.run(scheduler.run())

    def test_three_node_cycle_raises(self):
        """A → B → C → A should raise CircularDependencyError."""
        scheduler = TaskScheduler()
        t_a = make_task("A", dependencies=["C"])
        t_b = make_task("B", dependencies=["A"])
        t_c = make_task("C", dependencies=["B"])
        for t in (t_a, t_b, t_c):
            scheduler.add_task(t, factory())

        with pytest.raises(CircularDependencyError):
            asyncio.run(scheduler.run())

    def test_cycle_detected_in_get_execution_plan(self):
        """get_execution_plan should also raise CircularDependencyError."""
        scheduler = TaskScheduler()
        t_a = make_task("A", dependencies=["B"])
        t_b = make_task("B", dependencies=["A"])
        for t in (t_a, t_b):
            scheduler.add_task(t, factory())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_no_false_positive_for_diamond(self):
        """A diamond (A→B, A→C, B→D, C→D) is NOT a cycle."""
        scheduler = TaskScheduler()
        t_a = make_task("A")
        t_b = make_task("B", dependencies=["A"])
        t_c = make_task("C", dependencies=["A"])
        t_d = make_task("D", dependencies=["B", "C"])
        for t in (t_a, t_b, t_c, t_d):
            scheduler.add_task(t, factory())

        # Must not raise
        plan = scheduler.get_execution_plan()
        flat = [tid for wave in plan for tid in wave]
        assert flat.index("A") < flat.index("D")


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_task_retries_on_failure_then_succeeds(self):
        """A task that fails twice then succeeds should reach COMPLETED."""
        call_count = 0

        def flaky_factory():
            nonlocal call_count

            async def _coro():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("transient error")
                return "ok"

            return _coro

        scheduler = TaskScheduler(base_backoff=0.0)
        task = make_task("flaky", max_retries=3)
        scheduler.add_task(task, flaky_factory())

        results = asyncio.run(scheduler.run())

        assert results["flaky"].status == TaskStatus.COMPLETED
        assert results["flaky"].result == "ok"
        assert results["flaky"].retry_count == 2

    def test_task_fails_after_max_retries(self):
        """A task that always fails should be FAILED after exhausting retries."""
        def always_fail():
            async def _coro():
                raise RuntimeError("always")
            return _coro

        scheduler = TaskScheduler(base_backoff=0.0)
        task = make_task("bad", max_retries=2)
        scheduler.add_task(task, always_fail())

        results = asyncio.run(scheduler.run())

        assert results["bad"].status == TaskStatus.FAILED
        assert results["bad"].retry_count == 3  # initial + 2 retries

    def test_exponential_backoff_delays_increase(self):
        """Backoff delays should grow exponentially between retries."""
        delays_recorded: List[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float):
            delays_recorded.append(delay)
            # Skip actual waiting in tests
            return

        call_count = 0

        def flaky_factory():
            async def _coro():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RuntimeError("nope")
                return "done"
            return _coro

        scheduler = TaskScheduler(base_backoff=1.0)
        task = make_task("backoff_task", max_retries=3)
        scheduler.add_task(task, flaky_factory())

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(scheduler.run())

        # First retry: 1.0 * 2^0 = 1.0; second retry: 1.0 * 2^1 = 2.0
        assert len(delays_recorded) == 2
        assert delays_recorded[0] == pytest.approx(1.0)
        assert delays_recorded[1] == pytest.approx(2.0)

    def test_on_task_fail_event_emitted_per_final_failure(self):
        """on_task_fail should fire exactly once per ultimately-failed task."""
        fail_events: List[str] = []

        def always_fail():
            async def _coro():
                raise RuntimeError("x")
            return _coro

        scheduler = TaskScheduler(base_backoff=0.0)
        task = make_task("ev_fail", max_retries=1)
        scheduler.add_task(task, always_fail())

        async def on_fail(t: Task):
            fail_events.append(t.id)

        scheduler.on_task_fail(on_fail)
        asyncio.run(scheduler.run())

        assert fail_events.count("ev_fail") == 1


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    def test_concurrency_limit_respected(self):
        """No more than max_concurrency tasks should run simultaneously."""
        max_concurrency = 2
        current_running = 0
        peak_running = 0
        lock = asyncio.Lock()

        def tracked_factory(delay: float = 0.05):
            async def _coro():
                nonlocal current_running, peak_running
                async with lock:
                    current_running += 1
                    if current_running > peak_running:
                        peak_running = current_running
                await asyncio.sleep(delay)
                async with lock:
                    current_running -= 1
                return "done"
            return _coro

        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            t = make_task(f"c{i}")
            scheduler.add_task(t, tracked_factory())

        asyncio.run(scheduler.run())

        assert peak_running <= max_concurrency

    def test_independent_tasks_run_concurrently(self):
        """Independent tasks should overlap in time, not run sequentially."""
        DELAY = 0.05
        NUM_TASKS = 4

        scheduler = TaskScheduler(max_concurrency=NUM_TASKS)
        for i in range(NUM_TASKS):
            t = make_task(f"p{i}")
            scheduler.add_task(t, slow_factory(DELAY))

        start = time.monotonic()
        asyncio.run(scheduler.run())
        elapsed = time.monotonic() - start

        # If truly concurrent, elapsed ~ DELAY; sequential would be NUM_TASKS*DELAY
        assert elapsed < DELAY * NUM_TASKS * 0.75

    def test_concurrency_one_runs_sequentially(self):
        """With max_concurrency=1, tasks effectively run one at a time."""
        DELAY = 0.02
        NUM_TASKS = 3

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(NUM_TASKS):
            t = make_task(f"seq{i}")
            scheduler.add_task(t, slow_factory(DELAY))

        start = time.monotonic()
        asyncio.run(scheduler.run())
        elapsed = time.monotonic() - start

        # Must take at least NUM_TASKS * DELAY
        assert elapsed >= DELAY * NUM_TASKS * 0.9

    def test_on_task_start_called_for_every_task(self):
        """on_task_start should fire once per task."""
        started: List[str] = []

        async def handler(t: Task):
            started.append(t.id)

        scheduler = TaskScheduler(max_concurrency=3)
        ids = [f"s{i}" for i in range(5)]
        for tid in ids:
            scheduler.add_task(make_task(tid), factory())

        scheduler.on_task_start(handler)
        asyncio.run(scheduler.run())

        assert sorted(started) == sorted(ids)
