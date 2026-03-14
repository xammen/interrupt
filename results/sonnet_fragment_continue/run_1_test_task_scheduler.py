"""
test_task_scheduler.py
======================
Comprehensive pytest suite for task_scheduler.py.

Run with:
    pytest test_task_scheduler.py -v
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, call

import pytest

from task_scheduler import (
    CircularDependencyError,
    DuplicateTaskError,
    EventBus,
    MissingDependencyError,
    SchedulerMetrics,
    Task,
    TaskMetrics,
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


def success_fn(result=None):
    """Return a coroutine function that resolves immediately."""
    async def _coro():
        return result
    return _coro


def failing_fn(exc=RuntimeError("boom")):
    """Return a coroutine function that always raises."""
    async def _coro():
        raise exc
    return _coro


def fail_then_succeed(failures: int, result=None):
    """Return a coroutine function that fails *failures* times then succeeds."""
    calls = {"n": 0}

    async def _coro():
        if calls["n"] < failures:
            calls["n"] += 1
            raise RuntimeError("transient failure")
        return result

    return _coro


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_default_status_is_pending(self):
        t = make_task("t1")
        assert t.status == TaskStatus.PENDING

    def test_priority_validation_low(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=0)

    def test_priority_validation_high(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=11)

    def test_priority_boundary_values(self):
        t1 = make_task("a", priority=1)
        t10 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_created_at_is_utc(self):
        import datetime
        t = make_task("t")
        assert t.created_at.tzinfo == datetime.timezone.utc

    def test_result_defaults_none(self):
        assert make_task("t").result is None

    def test_dependencies_default_empty(self):
        assert make_task("t").dependencies == []


# ---------------------------------------------------------------------------
# TaskMetrics tests
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_when_not_started(self):
        m = TaskMetrics(task_id="t")
        assert m.elapsed is None

    def test_elapsed_none_when_not_finished(self):
        m = TaskMetrics(task_id="t", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed(self):
        m = TaskMetrics(task_id="t", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SchedulerMetrics tests
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_elapsed_none_when_not_started(self):
        assert SchedulerMetrics().total_elapsed is None

    def test_total_elapsed_computed(self):
        sm = SchedulerMetrics(total_start=0.0, total_end=10.0)
        assert sm.total_elapsed == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_calls_handler(self):
        bus = EventBus()
        handler = AsyncMock()
        task = make_task("t")
        bus.subscribe("on_task_start", handler)
        await bus.emit("on_task_start", task)
        handler.assert_awaited_once_with(task)

    @pytest.mark.asyncio
    async def test_emit_multiple_handlers(self):
        bus = EventBus()
        h1, h2 = AsyncMock(), AsyncMock()
        task = make_task("t")
        bus.subscribe("ev", h1)
        bus.subscribe("ev", h2)
        await bus.emit("ev", task)
        h1.assert_awaited_once_with(task)
        h2.assert_awaited_once_with(task)

    @pytest.mark.asyncio
    async def test_emit_unknown_event_no_error(self):
        bus = EventBus()
        # Should not raise.
        await bus.emit("unknown_event", make_task("t"))

    @pytest.mark.asyncio
    async def test_handlers_called_in_registration_order(self):
        order: List[int] = []
        bus = EventBus()

        async def h1(t):
            order.append(1)

        async def h2(t):
            order.append(2)

        bus.subscribe("ev", h1)
        bus.subscribe("ev", h2)
        await bus.emit("ev", make_task("t"))
        assert order == [1, 2]


# ---------------------------------------------------------------------------
# TaskScheduler — registration tests
# ---------------------------------------------------------------------------

class TestTaskSchedulerRegistration:
    def test_add_task_succeeds(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn())
        assert sched.get_task("t1") is not None

    def test_duplicate_task_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn())
        with pytest.raises(DuplicateTaskError):
            sched.add_task(make_task("t1"), success_fn())

    def test_get_task_returns_none_for_unknown(self):
        assert TaskScheduler().get_task("nope") is None

    def test_pending_tasks_all_after_registration(self):
        sched = TaskScheduler()
        for i in range(3):
            sched.add_task(make_task(f"t{i}"), success_fn())
        assert len(sched.pending_tasks()) == 3


# ---------------------------------------------------------------------------
# TaskScheduler — dependency validation
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    def test_missing_dependency_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1", dependencies=["t_missing"]), success_fn())
        with pytest.raises(MissingDependencyError):
            sched.get_execution_plan()

    def test_circular_dependency_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]), success_fn())
        sched.add_task(make_task("b", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_self_dependency_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_longer_cycle_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["c"]), success_fn())
        sched.add_task(make_task("b", dependencies=["a"]), success_fn())
        sched.add_task(make_task("c", dependencies=["b"]), success_fn())
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()


# ---------------------------------------------------------------------------
# TaskScheduler — execution plan
# ---------------------------------------------------------------------------

class TestExecutionPlan:
    def test_single_task_one_group(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn())
        plan = sched.get_execution_plan()
        assert plan == [["t1"]]

    def test_independent_tasks_one_group(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), success_fn())
        sched.add_task(make_task("b"), success_fn())
        plan = sched.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_linear_chain_separate_groups(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), success_fn())
        sched.add_task(make_task("b", dependencies=["a"]), success_fn())
        sched.add_task(make_task("c", dependencies=["b"]), success_fn())
        plan = sched.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_diamond_dependency(self):
        #        a
        #       / \
        #      b   c
        #       \ /
        #        d
        sched = TaskScheduler()
        sched.add_task(make_task("a"), success_fn())
        sched.add_task(make_task("b", dependencies=["a"]), success_fn())
        sched.add_task(make_task("c", dependencies=["a"]), success_fn())
        sched.add_task(make_task("d", dependencies=["b", "c"]), success_fn())
        plan = sched.get_execution_plan()
        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    def test_priority_ordering_within_group(self):
        sched = TaskScheduler()
        sched.add_task(make_task("low", priority=1), success_fn())
        sched.add_task(make_task("high", priority=9), success_fn())
        sched.add_task(make_task("mid", priority=5), success_fn())
        plan = sched.get_execution_plan()
        assert plan[0] == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# TaskScheduler — run (happy path)
# ---------------------------------------------------------------------------

class TestSchedulerRun:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn(result=42))
        await sched.run()
        assert sched.get_task("t1").status == TaskStatus.COMPLETED
        assert sched.get_task("t1").result == 42

    @pytest.mark.asyncio
    async def test_all_tasks_complete(self):
        sched = TaskScheduler()
        for i in range(5):
            sched.add_task(make_task(f"t{i}"), success_fn())
        await sched.run()
        assert all(
            sched.get_task(f"t{i}").status == TaskStatus.COMPLETED for i in range(5)
        )

    @pytest.mark.asyncio
    async def test_dependency_order_respected(self):
        execution_order: List[str] = []

        def ordered_fn(tid):
            async def _coro():
                execution_order.append(tid)
            return _coro

        sched = TaskScheduler()
        sched.add_task(make_task("a"), ordered_fn("a"))
        sched.add_task(make_task("b", dependencies=["a"]), ordered_fn("b"))
        sched.add_task(make_task("c", dependencies=["b"]), ordered_fn("c"))
        await sched.run()
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn())
        metrics = await sched.run()
        assert metrics.total_elapsed is not None
        assert metrics.total_elapsed >= 0
        assert "t1" in metrics.tasks
        assert metrics.tasks["t1"].elapsed is not None

    @pytest.mark.asyncio
    async def test_completed_tasks_list(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn())
        sched.add_task(make_task("t2"), success_fn())
        await sched.run()
        assert len(sched.completed_tasks()) == 2

    @pytest.mark.asyncio
    async def test_failed_tasks_list(self):
        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1", max_retries=0), failing_fn())
        await sched.run()
        assert len(sched.failed_tasks()) == 1

    @pytest.mark.asyncio
    async def test_state_reset_between_runs(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn(result="first"))
        await sched.run()
        # Re-register coroutine with different result (replace coroutine fn).
        sched._coroutines["t1"] = success_fn(result="second")
        await sched.run()
        assert sched.get_task("t1").result == "second"
        assert sched.get_task("t1").retry_count == 0


# ---------------------------------------------------------------------------
# TaskScheduler — retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self):
        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1", max_retries=2), failing_fn())
        await sched.run()
        task = sched.get_task("t1")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_task_succeeds_after_retries(self):
        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1", max_retries=3), fail_then_succeed(failures=2))
        await sched.run()
        task = sched.get_task("t1")
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self):
        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1", max_retries=0), failing_fn())
        await sched.run()
        task = sched.get_task("t1")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 1

    @pytest.mark.asyncio
    async def test_metrics_track_retry_count(self):
        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1", max_retries=2), fail_then_succeed(failures=1))
        metrics = await sched.run()
        assert metrics.tasks["t1"].retry_count == 1


# ---------------------------------------------------------------------------
# TaskScheduler — events
# ---------------------------------------------------------------------------

class TestEvents:
    @pytest.mark.asyncio
    async def test_on_task_start_fired(self):
        sched = TaskScheduler()
        handler = AsyncMock()
        sched.events.subscribe("on_task_start", handler)
        sched.add_task(make_task("t1"), success_fn())
        await sched.run()
        handler.assert_awaited_once()
        assert handler.call_args[0][0].id == "t1"

    @pytest.mark.asyncio
    async def test_on_task_complete_fired(self):
        sched = TaskScheduler()
        handler = AsyncMock()
        sched.events.subscribe("on_task_complete", handler)
        sched.add_task(make_task("t1"), success_fn())
        await sched.run()
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_task_fail_fired_each_attempt(self):
        sched = TaskScheduler(base_backoff=0.0)
        handler = AsyncMock()
        sched.events.subscribe("on_task_fail", handler)
        # 0 retries → 1 attempt → 1 failure event.
        sched.add_task(make_task("t1", max_retries=0), failing_fn())
        await sched.run()
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_on_task_fail_fired_per_retry(self):
        sched = TaskScheduler(base_backoff=0.0)
        handler = AsyncMock()
        sched.events.subscribe("on_task_fail", handler)
        # 2 retries → 3 attempts → 3 failure events.
        sched.add_task(make_task("t1", max_retries=2), failing_fn())
        await sched.run()
        assert handler.await_count == 3

    @pytest.mark.asyncio
    async def test_no_complete_event_on_failure(self):
        sched = TaskScheduler(base_backoff=0.0)
        complete_handler = AsyncMock()
        sched.events.subscribe("on_task_complete", complete_handler)
        sched.add_task(make_task("t1", max_retries=0), failing_fn())
        await sched.run()
        complete_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_tasks_all_events_fired(self):
        sched = TaskScheduler()
        starts, completes = AsyncMock(), AsyncMock()
        sched.events.subscribe("on_task_start", starts)
        sched.events.subscribe("on_task_complete", completes)
        for i in range(3):
            sched.add_task(make_task(f"t{i}"), success_fn())
        await sched.run()
        assert starts.await_count == 3
        assert completes.await_count == 3


# ---------------------------------------------------------------------------
# TaskScheduler — concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        """No more than max_concurrent tasks run simultaneously."""
        running = {"count": 0, "peak": 0}
        lock = asyncio.Lock()

        def counting_fn():
            async def _coro():
                async with lock:
                    running["count"] += 1
                    running["peak"] = max(running["peak"], running["count"])
                await asyncio.sleep(0.05)
                async with lock:
                    running["count"] -= 1
            return _coro

        sched = TaskScheduler(max_concurrent=2)
        for i in range(6):
            sched.add_task(make_task(f"t{i}"), counting_fn())
        await sched.run()
        assert running["peak"] <= 2

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Independent tasks finish faster together than sequentially."""
        sched = TaskScheduler(max_concurrent=4)

        async def slow():
            await asyncio.sleep(0.1)

        for i in range(4):
            sched.add_task(make_task(f"t{i}"), slow)

        import time
        start = time.monotonic()
        await sched.run()
        elapsed = time.monotonic() - start
        # Four 0.1 s tasks in parallel should finish well under 0.4 s.
        assert elapsed < 0.35


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_run(self):
        sched = TaskScheduler()
        metrics = await sched.run()
        assert metrics.total_elapsed is not None

    def test_execution_plan_empty_scheduler(self):
        sched = TaskScheduler()
        assert sched.get_execution_plan() == []

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), success_fn(result={"key": "value"}))
        await sched.run()
        assert sched.get_task("t1").result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_missing_dependency_caught_at_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1", dependencies=["ghost"]), success_fn())
        with pytest.raises(MissingDependencyError):
            await sched.run()

    def test_add_task_stores_coroutine(self):
        sched = TaskScheduler()
        fn = success_fn()
        sched.add_task(make_task("t1"), fn)
        assert sched._coroutines["t1"] is fn
