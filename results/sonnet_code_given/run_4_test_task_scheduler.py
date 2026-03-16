"""
test_task_scheduler.py - Comprehensive pytest test suite for task_scheduler.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
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
    dependencies: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def success_coro(task: Task) -> str:
    return f"result-{task.id}"


async def failing_coro(task: Task) -> None:
    raise RuntimeError("intentional failure")


def make_counted_coro(fail_times: int):
    """Returns a coroutine that fails `fail_times` times then succeeds."""
    call_count = {"n": 0}

    async def coro(task: Task) -> str:
        call_count["n"] += 1
        if call_count["n"] <= fail_times:
            raise RuntimeError(f"fail #{call_count['n']}")
        return "recovered"

    return coro


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_bounds(self):
        t = make_task("t1", priority=1)
        assert t.priority == 1
        t = make_task("t2", priority=10)
        assert t.priority == 10

    def test_invalid_priority_low(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=0)

    def test_invalid_priority_high(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t1")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_is_zero(self):
        t = make_task("t1")
        assert t.retry_count == 0

    def test_result_defaults_to_none(self):
        t = make_task("t1")
        assert t.result is None


# ---------------------------------------------------------------------------
# add_task / registration
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_task_registers_task(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, success_coro)
        assert scheduler.get_task("t1") is t

    def test_duplicate_id_raises(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, success_coro)
        with pytest.raises(ValueError, match="already registered"):
            scheduler.add_task(make_task("t1"), success_coro)

    def test_get_unknown_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            scheduler.get_task("nonexistent")


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, success_coro)
        await scheduler.run()
        assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_single_task_result_stored(self):
        scheduler = TaskScheduler()
        t = make_task("t1")
        scheduler.add_task(t, success_coro)
        await scheduler.run()
        assert t.result == "result-t1"

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        scheduler = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(5)]
        for t in tasks:
            scheduler.add_task(t, success_coro)
        await scheduler.run()
        assert all(t.status == TaskStatus.COMPLETED for t in tasks)

    @pytest.mark.asyncio
    async def test_completed_tasks_accessor(self):
        scheduler = TaskScheduler()
        for i in range(3):
            scheduler.add_task(make_task(f"t{i}"), success_coro)
        await scheduler.run()
        assert len(scheduler.completed_tasks()) == 3

    @pytest.mark.asyncio
    async def test_failed_tasks_accessor_empty_on_success(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), success_coro)
        await scheduler.run()
        assert scheduler.failed_tasks() == []

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), success_coro)
        metrics = await scheduler.run()
        assert isinstance(metrics, SchedulerMetrics)
        assert metrics.total_time is not None
        assert metrics.total_time >= 0

    @pytest.mark.asyncio
    async def test_per_task_metrics_elapsed(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), success_coro)
        metrics = await scheduler.run()
        assert metrics.per_task["t1"].elapsed is not None
        assert metrics.per_task["t1"].elapsed >= 0


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        order: list[str] = []

        async def record(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1, record)
        scheduler.add_task(t2, record)
        await scheduler.run()
        assert order.index("t1") < order.index("t2")

    @pytest.mark.asyncio
    async def test_chain_of_three_tasks_in_order(self):
        order: list[str] = []

        async def record(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t2"])
        for t in (t1, t2, t3):
            scheduler.add_task(t, record)
        await scheduler.run()
        assert order == ["t1", "t2", "t3"]

    @pytest.mark.asyncio
    async def test_diamond_dependency_all_complete(self):
        """t1 -> t2, t3 -> t4"""
        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t1"])
        t4 = make_task("t4", dependencies=["t2", "t3"])
        for t in (t1, t2, t3, t4):
            scheduler.add_task(t, success_coro)
        await scheduler.run()
        assert all(
            t.status == TaskStatus.COMPLETED for t in (t1, t2, t3, t4)
        )

    def test_unknown_dependency_raises_task_not_found(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1", dependencies=["ghost"]), success_coro)
        with pytest.raises(TaskNotFoundError):
            scheduler.get_execution_plan()

    def test_get_execution_plan_respects_layers(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1, success_coro)
        scheduler.add_task(t2, success_coro)
        plan = scheduler.get_execution_plan()
        # t1 must appear in an earlier group than t2
        layer_of = {}
        for idx, group in enumerate(plan):
            for tid in group:
                layer_of[tid] = idx
        assert layer_of["t1"] < layer_of["t2"]

    @pytest.mark.asyncio
    async def test_priority_influences_execution_order_within_layer(self):
        """Higher priority tasks should appear first in their layer."""
        order: list[str] = []

        async def record(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler(max_concurrency=1)
        t_low = make_task("low", priority=1)
        t_high = make_task("high", priority=9)
        scheduler.add_task(t_low, record)
        scheduler.add_task(t_high, record)
        await scheduler.run()
        assert order.index("high") < order.index("low")


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    def test_self_referential_task(self):
        scheduler = TaskScheduler()
        t = make_task("t1", dependencies=["t1"])
        scheduler.add_task(t, success_coro)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_task_cycle(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t2"])
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1, success_coro)
        scheduler.add_task(t2, success_coro)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_task_cycle(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t3"])
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t2"])
        for t in (t1, t2, t3):
            scheduler.add_task(t, success_coro)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1", dependencies=["t2"])
        t2 = make_task("t2", dependencies=["t1"])
        scheduler.add_task(t1, success_coro)
        scheduler.add_task(t2, success_coro)
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_no_false_positive_for_valid_dag(self):
        scheduler = TaskScheduler()
        t1 = make_task("t1")
        t2 = make_task("t2", dependencies=["t1"])
        t3 = make_task("t3", dependencies=["t1"])
        t4 = make_task("t4", dependencies=["t2", "t3"])
        for t in (t1, t2, t3, t4):
            scheduler.add_task(t, success_coro)
        # Should not raise
        plan = scheduler.get_execution_plan()
        assert plan  # non-empty


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_and_eventually_succeeds(
        self, monkeypatch
    ):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, make_counted_coro(fail_times=2))
        await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "recovered"
        assert t.retry_count == 2

    @pytest.mark.asyncio
    async def test_task_marked_failed_after_exceeding_max_retries(
        self, monkeypatch
    ):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=2)
        scheduler.add_task(t, failing_coro)
        await scheduler.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # max_retries + 1 attempts total

    @pytest.mark.asyncio
    async def test_failed_task_appears_in_failed_tasks(self, monkeypatch):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1", max_retries=0), failing_coro)
        await scheduler.run()
        failed = scheduler.failed_tasks()
        assert len(failed) == 1
        assert failed[0].id == "t1"

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self, monkeypatch):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=0)
        scheduler.add_task(t, failing_coro)
        await scheduler.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1

    @pytest.mark.asyncio
    async def test_metrics_reflect_retry_count(self, monkeypatch):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        scheduler = TaskScheduler()
        t = make_task("t1", max_retries=3)
        scheduler.add_task(t, make_counted_coro(fail_times=2))
        metrics = await scheduler.run()

        assert metrics.per_task["t1"].retry_count == 2


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At no point should more tasks be running than max_concurrency."""
        running = {"current": 0, "peak": 0}
        barrier = asyncio.Event()

        async def concurrent_coro(task: Task) -> None:
            running["current"] += 1
            running["peak"] = max(running["peak"], running["current"])
            # Yield control so other coroutines can run
            await asyncio.sleep(0)
            running["current"] -= 1

        max_concurrency = 2
        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), concurrent_coro)

        await scheduler.run()
        assert running["peak"] <= max_concurrency

    @pytest.mark.asyncio
    async def test_max_concurrency_one_serializes_all_tasks(self):
        order: list[str] = []
        active: list[bool] = [False]

        async def serial_coro(task: Task) -> None:
            assert not active[0], "Two tasks ran simultaneously!"
            active[0] = True
            await asyncio.sleep(0)
            order.append(task.id)
            active[0] = False

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), serial_coro)
        await scheduler.run()
        assert len(order) == 4

    @pytest.mark.asyncio
    async def test_independent_tasks_run_in_parallel_with_high_concurrency(self):
        """With concurrency >= task count, all independent tasks start before any finish."""
        started: list[str] = []
        finished: list[str] = []
        gate = asyncio.Event()

        async def gated_coro(task: Task) -> None:
            started.append(task.id)
            await gate.wait()
            finished.append(task.id)

        scheduler = TaskScheduler(max_concurrency=5)
        for i in range(5):
            scheduler.add_task(make_task(f"t{i}"), gated_coro)

        run_task = asyncio.ensure_future(scheduler.run())
        # Give all coroutines a chance to start and reach the gate
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(started) == 5, "All tasks should have started before the gate"
        gate.set()
        await run_task
        assert len(finished) == 5


# ---------------------------------------------------------------------------
# Observer / event callbacks
# ---------------------------------------------------------------------------

class TestObservers:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.add_task(make_task("t1"), success_coro)
        await scheduler.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        completed: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: completed.append(t.id))
        scheduler.add_task(make_task("t1"), success_coro)
        await scheduler.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called(self, monkeypatch):
        async def _noop_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        failed: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t: failed.append(t.id))
        scheduler.add_task(make_task("t1", max_retries=0), failing_coro)
        await scheduler.run()
        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_observer_exception_does_not_crash_scheduler(self):
        def bad_callback(task: Task) -> None:
            raise RuntimeError("observer error")

        scheduler = TaskScheduler()
        scheduler.on_task_complete(bad_callback)
        scheduler.add_task(make_task("t1"), success_coro)
        # Should not raise
        await scheduler.run()
        assert scheduler.get_task("t1").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_observers_all_called(self):
        calls: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: calls.append("obs1"))
        scheduler.on_task_complete(lambda t: calls.append("obs2"))
        scheduler.add_task(make_task("t1"), success_coro)
        await scheduler.run()
        assert "obs1" in calls
        assert "obs2" in calls
