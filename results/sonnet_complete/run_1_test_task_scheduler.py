"""
test_task_scheduler.py
======================
pytest test suite for task_scheduler.py.

Covers:
1. Basic task execution
2. Dependency resolution (topological ordering)
3. Circular dependency detection
4. Retry logic with exponential backoff
5. Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    DuplicateTaskError,
    MissingDependencyError,
    Task,
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
    dependencies: List[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


def succeed_fn(value: object = None):
    """Return an async callable that resolves to *value*."""
    async def _fn():
        return value
    return _fn


def fail_fn(exc: Exception | None = None):
    """Return an async callable that always raises *exc*."""
    async def _fn():
        raise exc or RuntimeError("intentional failure")
    return _fn


# ---------------------------------------------------------------------------
# Test 1 – Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, succeed_fn("hello"))

        results = await scheduler.run()

        assert task.status is TaskStatus.COMPLETED
        assert results["t1"] == "hello"
        assert task.result == "hello"

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        scheduler = TaskScheduler()
        for i in range(3):
            t = make_task(f"t{i}")
            scheduler.add_task(t, succeed_fn(i))

        results = await scheduler.run()

        for i in range(3):
            assert scheduler._tasks[f"t{i}"].status is TaskStatus.COMPLETED
            assert results[f"t{i}"] == i

    @pytest.mark.asyncio
    async def test_task_result_stored_on_task_object(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, succeed_fn(42))
        await scheduler.run()
        assert task.result == 42

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), succeed_fn())
        await scheduler.run()

        assert scheduler.metrics.total_elapsed is not None
        assert scheduler.metrics.total_elapsed >= 0
        m = scheduler.metrics.tasks["t1"]
        assert m.elapsed is not None and m.elapsed >= 0

    @pytest.mark.asyncio
    async def test_observer_on_task_complete_fires(self):
        completed: List[str] = []

        async def on_complete(task: Task) -> None:
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.events.subscribe("on_task_complete", on_complete)
        scheduler.add_task(make_task("t1"), succeed_fn())
        await scheduler.run()

        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), succeed_fn())
        with pytest.raises(DuplicateTaskError):
            scheduler.add_task(make_task("t1"), succeed_fn())

    @pytest.mark.asyncio
    async def test_task_priority_field_validated(self):
        with pytest.raises(ValueError):
            Task(id="bad", name="bad", priority=0)
        with pytest.raises(ValueError):
            Task(id="bad2", name="bad2", priority=11)


# ---------------------------------------------------------------------------
# Test 2 – Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        order: List[str] = []

        async def t1_fn():
            order.append("t1")

        async def t2_fn():
            order.append("t2")

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), t1_fn)
        scheduler.add_task(make_task("t2", dependencies=["t1"]), t2_fn)

        await scheduler.run()

        assert order.index("t1") < order.index("t2")

    @pytest.mark.asyncio
    async def test_execution_plan_groups_independent_tasks(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), succeed_fn())
        scheduler.add_task(make_task("b"), succeed_fn())
        scheduler.add_task(make_task("c", dependencies=["a", "b"]), succeed_fn())

        plan = scheduler.get_execution_plan()

        # "c" must be in a later group than both "a" and "b"
        assert len(plan) == 2
        first_group = set(plan[0])
        second_group = set(plan[1])
        assert {"a", "b"} == first_group
        assert {"c"} == second_group

    @pytest.mark.asyncio
    async def test_missing_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t2", dependencies=["t1"]), succeed_fn())
        with pytest.raises(MissingDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_deep_chain_executes_in_order(self):
        order: List[str] = []
        scheduler = TaskScheduler()

        scheduler.add_task(make_task("a"), succeed_fn())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeed_fn())
        scheduler.add_task(make_task("c", dependencies=["b"]), succeed_fn())

        async def on_start(task: Task) -> None:
            order.append(task.id)

        scheduler.events.subscribe("on_task_start", on_start)
        await scheduler.run()

        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_priority_ordering_within_wave(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("low", priority=1), succeed_fn())
        scheduler.add_task(make_task("high", priority=9), succeed_fn())
        scheduler.add_task(make_task("mid", priority=5), succeed_fn())

        plan = scheduler.get_execution_plan()
        # Single wave; highest priority first
        assert plan[0][0] == "high"
        assert plan[0][-1] == "low"


# ---------------------------------------------------------------------------
# Test 3 – Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    def test_simple_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), succeed_fn())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeed_fn())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]), succeed_fn())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]), succeed_fn())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeed_fn())
        scheduler.add_task(make_task("c", dependencies=["b"]), succeed_fn())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_propagates_circular_dependency_error(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("x", dependencies=["y"]), succeed_fn())
        scheduler.add_task(make_task("y", dependencies=["x"]), succeed_fn())
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_no_cycle_does_not_raise(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), succeed_fn())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeed_fn())
        # Should not raise
        plan = scheduler.get_execution_plan()
        assert len(plan) == 2


# ---------------------------------------------------------------------------
# Test 4 – Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_then_succeeds(self):
        """Fail twice, succeed on third attempt."""
        attempts = {"count": 0}

        async def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("not yet")
            return "ok"

        scheduler = TaskScheduler(base_backoff=0.0)  # no real delay in tests
        task = make_task("t1", max_retries=3)
        scheduler.add_task(task, flaky)

        results = await scheduler.run()

        assert task.status is TaskStatus.COMPLETED
        assert results["t1"] == "ok"
        assert task.retry_count == 2  # two retries before success
        assert attempts["count"] == 3

    @pytest.mark.asyncio
    async def test_task_marked_failed_after_max_retries(self):
        scheduler = TaskScheduler(base_backoff=0.0)
        task = make_task("t1", max_retries=2)
        scheduler.add_task(task, fail_fn())

        with pytest.raises(RuntimeError):
            await scheduler.run()

        assert task.status is TaskStatus.FAILED
        assert task.retry_count == 2

    @pytest.mark.asyncio
    async def test_on_task_fail_event_fires(self):
        failed: List[str] = []

        async def on_fail(task: Task) -> None:
            failed.append(task.id)

        scheduler = TaskScheduler(base_backoff=0.0)
        scheduler.events.subscribe("on_task_fail", on_fail)
        task = make_task("t1", max_retries=0)
        scheduler.add_task(task, fail_fn())

        with pytest.raises(RuntimeError):
            await scheduler.run()

        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays_increase(self):
        """Verify that sleep durations double with each retry attempt."""
        sleep_calls: List[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempts = {"count": 0}

        async def always_fail():
            attempts["count"] += 1
            raise RuntimeError("boom")

        scheduler = TaskScheduler(base_backoff=1.0)
        task = make_task("t1", max_retries=3)
        scheduler.add_task(task, always_fail)

        with patch("task_scheduler.asyncio.sleep", side_effect=_fake_sleep):
            with pytest.raises(RuntimeError):
                await scheduler.run()

        # For max_retries=3 there are 3 sleep calls (before attempts 2, 3, 4)
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self):
        attempts = {"count": 0}

        async def fn():
            attempts["count"] += 1
            raise ValueError("nope")

        scheduler = TaskScheduler(base_backoff=0.0)
        task = make_task("t1", max_retries=0)
        scheduler.add_task(task, fn)

        with pytest.raises(ValueError):
            await scheduler.run()

        assert attempts["count"] == 1
        assert task.retry_count == 0


# ---------------------------------------------------------------------------
# Test 5 – Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At most `max_concurrent` tasks should run at the same time."""
        concurrency_limit = 2
        running: list[int] = []
        peak: list[int] = []
        lock = asyncio.Lock()

        async def counted_task():
            async with lock:
                running.append(1)
                peak.append(len(running))

            await asyncio.sleep(0.05)  # simulate work

            async with lock:
                running.pop()

        scheduler = TaskScheduler(max_concurrent=concurrency_limit)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), counted_task)

        await scheduler.run()

        assert max(peak) <= concurrency_limit

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """With limit > task count, all tasks in a wave should overlap."""
        started: List[float] = []
        finished: List[float] = []

        async def slow():
            started.append(time.monotonic())
            await asyncio.sleep(0.05)
            finished.append(time.monotonic())

        scheduler = TaskScheduler(max_concurrent=10)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), slow)

        await scheduler.run()

        # The first task should finish *after* the last task started
        # (i.e. they overlapped in time).
        assert min(finished) > min(started)
        total = max(finished) - min(started)
        # 4 tasks * 50 ms each serially = 200 ms; concurrent should be ~50 ms
        assert total < 0.18, f"Tasks appear to have run serially: {total:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrency_limit_one_runs_serially(self):
        """With max_concurrent=1 tasks within a single wave run one-by-one."""
        timeline: List[tuple[float, float]] = []

        async def timed():
            start = time.monotonic()
            await asyncio.sleep(0.03)
            timeline.append((start, time.monotonic()))

        scheduler = TaskScheduler(max_concurrent=1)
        for i in range(3):
            scheduler.add_task(make_task(f"t{i}"), timed)

        await scheduler.run()

        # No two intervals should overlap
        sorted_tl = sorted(timeline)
        for i in range(1, len(sorted_tl)):
            prev_end = sorted_tl[i - 1][1]
            cur_start = sorted_tl[i][0]
            assert cur_start >= prev_end - 1e-4, (
                f"Tasks overlapped: task {i} started at {cur_start:.4f} "
                f"but previous ended at {prev_end:.4f}"
            )

    @pytest.mark.asyncio
    async def test_on_task_start_fires_for_each_task(self):
        started_ids: List[str] = []

        async def on_start(task: Task) -> None:
            started_ids.append(task.id)

        scheduler = TaskScheduler(max_concurrent=5)
        scheduler.events.subscribe("on_task_start", on_start)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), succeed_fn())

        await scheduler.run()

        assert sorted(started_ids) == ["t0", "t1", "t2", "t3"]
