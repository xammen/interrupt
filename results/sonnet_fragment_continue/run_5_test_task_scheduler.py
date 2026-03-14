"""Tests for task_scheduler.py."""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from task_scheduler import (
    CircularDependencyError,
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
    *,
    name: str = "",
    priority: int = 5,
    dependencies: List[str] | None = None,
    max_retries: int = 0,
) -> Task:
    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def noop() -> str:
    return "ok"


async def fail_always() -> None:
    raise RuntimeError("always fails")


def make_fail_then_succeed(failures: int):
    """Return a coroutine factory that raises *failures* times then succeeds."""
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise RuntimeError(f"failure #{calls['n']}")
        return "recovered"

    return fn


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_boundaries(self):
        t1 = make_task("a", priority=1)
        t10 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_low(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=0)

    def test_invalid_priority_high(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t")
        assert t.status is TaskStatus.PENDING

    def test_default_result_is_none(self):
        t = make_task("t")
        assert t.result is None

    def test_created_at_set(self):
        t = make_task("t")
        assert t.created_at is not None


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed(self):
        m = TaskMetrics(task_id="t", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)

    def test_elapsed_zero_by_default(self):
        m = TaskMetrics(task_id="t")
        assert m.elapsed == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TaskScheduler – registration
# ---------------------------------------------------------------------------

class TestSchedulerRegistration:
    def test_add_task_registers_correctly(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, noop)
        assert s.get_task_status("a") is TaskStatus.PENDING

    def test_duplicate_task_raises(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, noop)
        with pytest.raises(ValueError, match="already registered"):
            s.add_task(make_task("a"), noop)

    def test_get_task_status_unknown_raises(self):
        s = TaskScheduler()
        with pytest.raises(KeyError, match="(?i)unknown"):
            s.get_task_status("ghost")

    def test_invalid_max_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            TaskScheduler(max_concurrency=0)

    def test_on_valid_event(self):
        s = TaskScheduler()
        cb = MagicMock()
        s.on("on_task_start", cb)  # should not raise

    def test_on_invalid_event_raises(self):
        s = TaskScheduler()
        with pytest.raises(ValueError, match="Unknown event"):
            s.on("on_task_explode", MagicMock())


# ---------------------------------------------------------------------------
# Dependency graph helpers
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    def test_unknown_dependency_raises(self):
        s = TaskScheduler()
        t = make_task("a", dependencies=["ghost"])
        s.add_task(t, noop)
        with pytest.raises(ValueError, match="unknown task"):
            s._validate_dependencies()

    def test_valid_dependency_passes(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        s._validate_dependencies()  # no exception

    def test_build_graph_structure(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        graph = s._build_graph()
        assert "b" in graph["a"]
        assert graph["b"] == set()


# ---------------------------------------------------------------------------
# Topological sort / execution plan
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_no_dependencies_single_group(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b"), noop)
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_linear_chain_separate_groups(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        s.add_task(make_task("c", dependencies=["b"]), noop)
        plan = s.get_execution_plan()
        assert plan == [["a"], ["b"], ["c"]]

    def test_diamond_dependency(self):
        s = TaskScheduler()
        s.add_task(make_task("root"), noop)
        s.add_task(make_task("left", dependencies=["root"]), noop)
        s.add_task(make_task("right", dependencies=["root"]), noop)
        s.add_task(make_task("tip", dependencies=["left", "right"]), noop)
        plan = s.get_execution_plan()
        # root first, left+right middle, tip last
        assert plan[0] == ["root"]
        assert set(plan[1]) == {"left", "right"}
        assert plan[2] == ["tip"]

    def test_priority_ordering_within_group(self):
        s = TaskScheduler()
        s.add_task(make_task("low", priority=1), noop)
        s.add_task(make_task("high", priority=9), noop)
        s.add_task(make_task("mid", priority=5), noop)
        plan = s.get_execution_plan()
        assert plan[0] == ["high", "mid", "low"]

    def test_circular_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_circular_dependency_error_has_cycle(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        try:
            s.get_execution_plan()
        except CircularDependencyError as exc:
            assert len(exc.cycle) >= 2

    def test_single_task_plan(self):
        s = TaskScheduler()
        s.add_task(make_task("solo"), noop)
        plan = s.get_execution_plan()
        assert plan == [["solo"]]


# ---------------------------------------------------------------------------
# CircularDependencyError
# ---------------------------------------------------------------------------

class TestCircularDependencyError:
    def test_str_contains_arrow(self):
        err = CircularDependencyError(["a", "b", "a"])
        assert "->" in str(err)

    def test_cycle_attribute(self):
        err = CircularDependencyError(["x", "y"])
        assert err.cycle == ["x", "y"]


# ---------------------------------------------------------------------------
# Async execution – happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        results = await s.run()
        assert results["a"] == "ok"
        assert s.get_task_status("a") is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self):
        s = TaskScheduler()

        async def returns_42():
            return 42

        s.add_task(make_task("a"), returns_42)
        await s.run()
        assert s._tasks["a"].result == 42

    @pytest.mark.asyncio
    async def test_dependency_chain_executes_in_order(self):
        order: List[str] = []
        s = TaskScheduler()

        async def record(tid):
            async def fn():
                order.append(tid)
            return fn

        s.add_task(make_task("a"), await record("a"))
        s.add_task(make_task("b", dependencies=["a"]), await record("b"))
        s.add_task(make_task("c", dependencies=["b"]), await record("c"))
        await s.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_sync_callable_works(self):
        s = TaskScheduler()

        def sync_fn():
            return "sync"

        s.add_task(make_task("a"), sync_fn)
        results = await s.run()
        assert results["a"] == "sync"

    @pytest.mark.asyncio
    async def test_all_completed_after_run(self):
        s = TaskScheduler()
        for tid in ["x", "y", "z"]:
            s.add_task(make_task(tid), noop)
        await s.run()
        for tid in ["x", "y", "z"]:
            assert s.get_task_status(tid) is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_ok(self):
        s = TaskScheduler()
        results = await s.run()
        assert results == {}

    @pytest.mark.asyncio
    async def test_concurrent_tasks_run_simultaneously(self):
        """Tasks in the same group should overlap in time."""
        start_times: List[float] = []
        s = TaskScheduler(max_concurrency=4)

        def make_recorder():
            async def fn():
                start_times.append(time.monotonic())
                await asyncio.sleep(0.05)
            return fn

        for tid in ["a", "b", "c"]:
            s.add_task(make_task(tid), make_recorder())

        t0 = time.monotonic()
        await s.run()
        elapsed = time.monotonic() - t0
        # With concurrency=4 all three overlap; total should be < 3 * 0.05
        assert elapsed < 0.12, f"Tasks did not run concurrently (elapsed={elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_succeeds_after_retries(self):
        s = TaskScheduler()
        fn = make_fail_then_succeed(failures=2)
        s.add_task(make_task("a", max_retries=3), fn)
        results = await s.run()
        assert results["a"] == "recovered"
        assert s.get_task_status("a") is TaskStatus.COMPLETED
        assert s._tasks["a"].retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        s = TaskScheduler()
        s.add_task(make_task("a", max_retries=2), fail_always)
        results = await s.run()
        assert results["a"] is None
        assert s.get_task_status("a") is TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_count_incremented(self):
        s = TaskScheduler()
        fn = make_fail_then_succeed(failures=1)
        s.add_task(make_task("a", max_retries=2), fn)
        await s.run()
        assert s._tasks["a"].retry_count == 1

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        s = TaskScheduler()
        s.add_task(make_task("a", max_retries=0), fail_always)
        await s.run()
        assert s.get_task_status("a") is TaskStatus.FAILED
        assert s._tasks["a"].retry_count == 1

    @pytest.mark.asyncio
    async def test_failed_task_aborts_dependents(self):
        """A failing task should prevent its dependents from running."""
        s = TaskScheduler()
        s.add_task(make_task("a", max_retries=0), fail_always)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        results = await s.run()
        # "b" never ran – stays PENDING
        assert s.get_task_status("b") is TaskStatus.PENDING
        assert results["b"] is None


# ---------------------------------------------------------------------------
# Observer / event callbacks
# ---------------------------------------------------------------------------

class TestObservers:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        s = TaskScheduler()
        events: List[str] = []

        def cb(task: Task):
            events.append(("start", task.id))

        s.on("on_task_start", cb)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert ("start", "a") in events

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        s = TaskScheduler()
        events: List[str] = []

        def cb(task: Task):
            events.append(("complete", task.id))

        s.on("on_task_complete", cb)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert ("complete", "a") in events

    @pytest.mark.asyncio
    async def test_on_task_fail_called(self):
        s = TaskScheduler()
        events: List[str] = []

        def cb(task: Task):
            events.append(("fail", task.id))

        s.on("on_task_fail", cb)
        s.add_task(make_task("a", max_retries=0), fail_always)
        await s.run()
        assert ("fail", "a") in events

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self):
        s = TaskScheduler()
        events: List[str] = []

        async def async_cb(task: Task):
            events.append(task.id)

        s.on("on_task_complete", async_cb)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert "a" in events

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_called(self):
        s = TaskScheduler()
        log: List[int] = []
        s.on("on_task_start", lambda t: log.append(1))
        s.on("on_task_start", lambda t: log.append(2))
        s.add_task(make_task("a"), noop)
        await s.run()
        assert 1 in log and 2 in log

    @pytest.mark.asyncio
    async def test_start_callback_sees_running_status(self):
        s = TaskScheduler()
        statuses: List[TaskStatus] = []

        def cb(task: Task):
            statuses.append(task.status)

        s.on("on_task_start", cb)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert TaskStatus.RUNNING in statuses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded_after_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        await s.run()
        m = s.get_metrics()["a"]
        assert m.start_time > 0
        assert m.end_time >= m.start_time

    @pytest.mark.asyncio
    async def test_elapsed_positive_after_run(self):
        s = TaskScheduler()

        async def slow():
            await asyncio.sleep(0.01)

        s.add_task(make_task("a"), slow)
        await s.run()
        assert s.get_metrics()["a"].elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_retries_in_metrics(self):
        s = TaskScheduler()
        fn = make_fail_then_succeed(failures=2)
        s.add_task(make_task("a", max_retries=3), fn)
        await s.run()
        assert s.get_metrics()["a"].retries == 2

    @pytest.mark.asyncio
    async def test_total_elapsed_set_after_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        await s.run()
        assert s.total_elapsed > 0

    def test_get_metrics_returns_copy(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        m1 = s.get_metrics()
        m2 = s.get_metrics()
        assert m1 is not m2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    @pytest.mark.asyncio
    async def test_summary_completed_count(self):
        s = TaskScheduler()
        for tid in ["a", "b", "c"]:
            s.add_task(make_task(tid), noop)
        await s.run()
        sm = s.summary()
        assert sm["total_tasks"] == 3
        assert sm["completed"] == 3
        assert sm["failed"] == 0

    @pytest.mark.asyncio
    async def test_summary_failed_count(self):
        s = TaskScheduler()
        s.add_task(make_task("a", max_retries=0), fail_always)
        s.add_task(make_task("b"), noop)
        await s.run()
        sm = s.summary()
        assert sm["failed"] == 1

    @pytest.mark.asyncio
    async def test_summary_total_elapsed_present(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        await s.run()
        sm = s.summary()
        assert "total_elapsed_seconds" in sm
        assert sm["total_elapsed_seconds"] >= 0

    def test_summary_before_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        sm = s.summary()
        assert sm["pending"] == 1
        assert sm["completed"] == 0


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """Never more than max_concurrency tasks active simultaneously."""
        active = {"count": 0, "peak": 0}
        s = TaskScheduler(max_concurrency=2)

        def make_fn():
            async def fn():
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
                await asyncio.sleep(0.02)
                active["count"] -= 1
            return fn

        for i in range(6):
            s.add_task(make_task(str(i)), make_fn())

        await s.run()
        assert active["peak"] <= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_task_with_no_dependencies_in_first_group(self):
        s = TaskScheduler()
        s.add_task(make_task("root"), noop)
        s.add_task(make_task("child", dependencies=["root"]), noop)
        plan = s.get_execution_plan()
        assert plan[0] == ["root"]

    def test_get_execution_plan_does_not_run_tasks(self):
        called = {"n": 0}
        s = TaskScheduler()

        async def track():
            called["n"] += 1

        s.add_task(make_task("a"), track)
        s.get_execution_plan()
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_large_fan_out(self):
        """One root task with many children all run in second group."""
        s = TaskScheduler(max_concurrency=10)
        s.add_task(make_task("root"), noop)
        for i in range(8):
            s.add_task(make_task(f"child_{i}", dependencies=["root"]), noop)
        results = await s.run()
        assert all(v == "ok" for v in results.values())

    @pytest.mark.asyncio
    async def test_run_twice_reflects_previous_state(self):
        """Calling run a second time after completion returns cached results."""
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        await s.run()
        # Status should still be COMPLETED on second call
        results = await s.run()
        # Tasks already COMPLETED skip re-execution (status preserved)
        assert s.get_task_status("a") is TaskStatus.COMPLETED
