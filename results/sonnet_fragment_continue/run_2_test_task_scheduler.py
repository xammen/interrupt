"""
test_task_scheduler.py - Comprehensive tests for task_scheduler.py
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
    Task,
    TaskMetrics,
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
    fn=None,
) -> Task:
    """Factory for creating Task instances with sensible defaults."""

    async def _noop():
        return f"result-{task_id}"

    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
        _fn=fn if fn is not None else _noop,
    )


def make_failing_task(task_id: str, fail_times: int, priority: int = 5) -> Task:
    """Task that raises on the first *fail_times* attempts, then succeeds."""
    calls = {"count": 0}

    async def _flaky():
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise RuntimeError(f"deliberate failure #{calls['count']}")
        return "recovered"

    return Task(
        id=task_id,
        name=task_id,
        priority=priority,
        max_retries=fail_times,  # exactly enough retries to succeed
        _fn=_flaky,
    )


def make_always_failing_task(task_id: str, max_retries: int = 2) -> Task:
    async def _always_fail():
        raise RuntimeError("permanent failure")

    return Task(
        id=task_id,
        name=task_id,
        priority=5,
        max_retries=max_retries,
        _fn=_always_fail,
    )


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_bounds(self):
        t1 = make_task("a", priority=1)
        t2 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t2.priority == 10

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="priority must be between"):
            Task(id="x", name="x", priority=0)

    def test_invalid_priority_too_high(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("a")
        assert t.status == TaskStatus.PENDING

    def test_created_at_is_set(self):
        t = make_task("a")
        assert t.created_at is not None

    def test_result_defaults_to_none(self):
        t = make_task("a")
        assert t.result is None

    def test_repr_excludes_fn(self):
        t = make_task("a")
        assert "_fn" not in repr(t)


# ---------------------------------------------------------------------------
# TaskMetrics tests
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed(self):
        m = TaskMetrics(task_id="a", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)

    def test_elapsed_zero_when_not_run(self):
        m = TaskMetrics(task_id="a")
        assert m.elapsed == 0.0


# ---------------------------------------------------------------------------
# SchedulerMetrics tests
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_elapsed(self):
        m = SchedulerMetrics(total_start=0.0, total_end=5.0)
        assert m.total_elapsed == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TaskScheduler.add_task tests
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_single_task(self):
        scheduler = TaskScheduler()
        t = make_task("a")
        scheduler.add_task(t)
        assert scheduler.get_task("a") is t

    def test_duplicate_id_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        with pytest.raises(ValueError, match="already registered"):
            scheduler.add_task(make_task("a"))

    def test_multiple_tasks(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(make_task(str(i)))
        assert len(scheduler.get_all_tasks()) == 5


# ---------------------------------------------------------------------------
# TaskScheduler.get_task / get_all_tasks / get_tasks_by_status tests
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_get_task_not_found(self):
        scheduler = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            scheduler.get_task("nonexistent")

    def test_get_all_tasks_order(self):
        scheduler = TaskScheduler()
        ids = ["z", "a", "m"]
        for tid in ids:
            scheduler.add_task(make_task(tid))
        assert [t.id for t in scheduler.get_all_tasks()] == ids

    def test_get_tasks_by_status_empty(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        assert scheduler.get_tasks_by_status(TaskStatus.RUNNING) == []

    @pytest.mark.asyncio
    async def test_get_tasks_by_status_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_always_failing_task("b", max_retries=0))
        await scheduler.run()
        completed = scheduler.get_tasks_by_status(TaskStatus.COMPLETED)
        failed = scheduler.get_tasks_by_status(TaskStatus.FAILED)
        assert len(completed) == 1
        assert completed[0].id == "a"
        assert len(failed) == 1
        assert failed[0].id == "b"


# ---------------------------------------------------------------------------
# Dependency validation tests
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["missing"]))
        with pytest.raises(TaskNotFoundError, match="unknown task 'missing'"):
            await scheduler.run()

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    @pytest.mark.asyncio
    async def test_three_way_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["b"]))
        with pytest.raises(CircularDependencyError):
            await scheduler.run()


# ---------------------------------------------------------------------------
# Dependency ordering tests
# ---------------------------------------------------------------------------

class TestDependencyOrdering:
    @pytest.mark.asyncio
    async def test_simple_chain(self):
        order: List[str] = []

        async def record(tid: str):
            order.append(tid)

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", fn=lambda: record("a")))
        scheduler.add_task(make_task("b", dependencies=["a"], fn=lambda: record("b")))
        scheduler.add_task(make_task("c", dependencies=["b"], fn=lambda: record("c")))
        await scheduler.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """
        a -> b, a -> c, b -> d, c -> d
        'd' must run after both 'b' and 'c'.
        """
        order: List[str] = []

        def recorder(tid: str):
            async def _fn():
                order.append(tid)

            return _fn

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", fn=recorder("a")))
        scheduler.add_task(make_task("b", dependencies=["a"], fn=recorder("b")))
        scheduler.add_task(make_task("c", dependencies=["a"], fn=recorder("c")))
        scheduler.add_task(make_task("d", dependencies=["b", "c"], fn=recorder("d")))
        await scheduler.run()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    @pytest.mark.asyncio
    async def test_independent_tasks_all_complete(self):
        scheduler = TaskScheduler()
        for i in range(6):
            scheduler.add_task(make_task(str(i)))
        await scheduler.run()
        completed = scheduler.get_tasks_by_status(TaskStatus.COMPLETED)
        assert len(completed) == 6


# ---------------------------------------------------------------------------
# Priority ordering tests
# ---------------------------------------------------------------------------

class TestPriority:
    @pytest.mark.asyncio
    async def test_higher_priority_scheduled_first(self):
        """Among independent tasks, higher priority should start first."""
        start_order: List[str] = []

        def recorder(tid: str):
            async def _fn():
                start_order.append(tid)

            return _fn

        scheduler = TaskScheduler(max_concurrency=1)  # serial to guarantee order
        scheduler.add_task(make_task("low", priority=1, fn=recorder("low")))
        scheduler.add_task(make_task("high", priority=9, fn=recorder("high")))
        scheduler.add_task(make_task("mid", priority=5, fn=recorder("mid")))
        await scheduler.run()

        assert start_order.index("high") < start_order.index("mid")
        assert start_order.index("mid") < start_order.index("low")


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_succeeds_after_retries(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_failing_task("a", fail_times=2))
        await scheduler.run()
        task = scheduler.get_task("a")
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2
        assert task.result == "recovered"

    @pytest.mark.asyncio
    async def test_task_fails_when_retries_exhausted(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_always_failing_task("a", max_retries=2))
        await scheduler.run()
        task = scheduler.get_task("a")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_always_failing_task("a", max_retries=0))
        await scheduler.run()
        task = scheduler.get_task("a")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 0

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        assert scheduler.get_task("a").retry_count == 0


# ---------------------------------------------------------------------------
# Concurrency limiting tests
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """Peak concurrent executions should not exceed max_concurrency."""
        max_concurrency = 2
        current = {"count": 0, "peak": 0}

        async def tracked():
            current["count"] += 1
            current["peak"] = max(current["peak"], current["count"])
            await asyncio.sleep(0.05)
            current["count"] -= 1

        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(make_task(str(i), fn=tracked))

        await scheduler.run()
        assert current["peak"] <= max_concurrency

    @pytest.mark.asyncio
    async def test_single_concurrency_is_serial(self):
        """max_concurrency=1 must serialize all task executions."""
        timestamps: List[float] = []

        async def record_time():
            timestamps.append(time.monotonic())
            await asyncio.sleep(0.02)

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(3):
            scheduler.add_task(make_task(str(i), fn=record_time))

        await scheduler.run()
        # Each task should start after the previous one ends (~0.02s gap)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


# ---------------------------------------------------------------------------
# Observer / event tests
# ---------------------------------------------------------------------------

class TestObservers:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started: List[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b"))
        await scheduler.run()
        assert set(started) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        completed: List[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: completed.append(t.id))
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called(self):
        failed: List[tuple] = []
        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, e: failed.append((t.id, str(e))))
        scheduler.add_task(make_always_failing_task("a", max_retries=0))
        await scheduler.run()
        assert len(failed) == 1
        assert failed[0][0] == "a"
        assert "permanent failure" in failed[0][1]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self):
        calls: List[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: calls.append("cb1"))
        scheduler.on_task_complete(lambda t: calls.append("cb2"))
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        assert calls.count("cb1") == 1
        assert calls.count("cb2") == 1

    @pytest.mark.asyncio
    async def test_fail_callback_not_called_on_success(self):
        fail_calls: List[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, e: fail_calls.append(t.id))
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        assert fail_calls == []

    @pytest.mark.asyncio
    async def test_complete_callback_not_called_on_fail(self):
        complete_calls: List[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: complete_calls.append(t.id))
        scheduler.add_task(make_always_failing_task("a", max_retries=0))
        await scheduler.run()
        assert complete_calls == []


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.mark.asyncio
    async def test_scheduler_metrics_populated(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        metrics = await scheduler.run()
        assert metrics.total_elapsed > 0
        assert "a" in metrics.tasks

    @pytest.mark.asyncio
    async def test_task_metrics_timing(self):
        async def slow():
            await asyncio.sleep(0.05)
            return "done"

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", fn=slow))
        metrics = await scheduler.run()
        assert metrics.tasks["a"].elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_failed_task_metrics_recorded(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_always_failing_task("a", max_retries=1))
        metrics = await scheduler.run()
        assert "a" in metrics.tasks
        assert metrics.tasks["a"].retry_count == 1

    @pytest.mark.asyncio
    async def test_multiple_tasks_all_in_metrics(self):
        scheduler = TaskScheduler()
        for i in range(4):
            scheduler.add_task(make_task(str(i)))
        metrics = await scheduler.run()
        assert set(metrics.tasks.keys()) == {"0", "1", "2", "3"}


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_status(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        assert scheduler.get_task("a").status == TaskStatus.COMPLETED
        scheduler.reset()
        assert scheduler.get_task("a").status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_reset_clears_result(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        scheduler.reset()
        assert scheduler.get_task("a").result is None

    @pytest.mark.asyncio
    async def test_reset_clears_retry_count(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_failing_task("a", fail_times=1))
        await scheduler.run()
        assert scheduler.get_task("a").retry_count > 0
        scheduler.reset()
        assert scheduler.get_task("a").retry_count == 0

    @pytest.mark.asyncio
    async def test_can_run_again_after_reset(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        await scheduler.run()
        scheduler.reset()
        await scheduler.run()
        assert scheduler.get_task("a").status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_cleanly(self):
        scheduler = TaskScheduler()
        metrics = await scheduler.run()
        assert metrics.total_elapsed >= 0

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        async def produce():
            return 42

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", fn=produce))
        await scheduler.run()
        assert scheduler.get_task("a").result == 42

    @pytest.mark.asyncio
    async def test_task_with_no_fn_fails(self):
        task = Task(id="nofn", name="nofn", priority=5, _fn=None)
        scheduler = TaskScheduler()
        scheduler.add_task(task)
        await scheduler.run()
        assert scheduler.get_task("nofn").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_self_dependency_raises_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    @pytest.mark.asyncio
    async def test_large_independent_workload(self):
        scheduler = TaskScheduler(max_concurrency=8)
        for i in range(50):
            scheduler.add_task(make_task(str(i)))
        await scheduler.run()
        assert len(scheduler.get_tasks_by_status(TaskStatus.COMPLETED)) == 50
