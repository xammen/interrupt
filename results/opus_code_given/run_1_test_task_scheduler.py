"""
Comprehensive tests for task_scheduler.py.

Covers:
  - Task dataclass creation, defaults, and validation
  - TaskMetrics / SchedulerMetrics
  - Basic single & multi-task execution
  - Dependency resolution (linear chain, diamond, multi-tier, priority ordering)
  - Circular dependency detection (self-loop, 2-node, 3-node cycles)
  - Retry logic with exponential backoff
  - Concurrency-limit enforcement
  - Observer / event system (sync & async callbacks)
  - Edge cases (duplicate IDs, unknown deps, failed-dep cascading, accessors)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop(task: Task) -> str:
    """Simple coroutine that succeeds immediately."""
    return f"{task.id}-done"


async def _slow(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.id}-done"


def _make_failing_coro(*, fail_times: int = 1):
    """Return an async callable that raises on the first *fail_times* calls,
    then succeeds.
    """
    call_count = 0

    async def _coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"transient error #{call_count}")
        return f"{task.id}-recovered"

    return _coro


def _make_always_failing_coro():
    """Return an async callable that always raises."""
    async def _coro(task: Task) -> str:
        raise RuntimeError("permanent failure")
    return _coro


# ===================================================================
# Task dataclass
# ===================================================================

class TestTaskDataclass:
    """Tests for the Task dataclass construction and validation."""

    def test_create_with_defaults(self):
        t = Task(id="t1", name="Task 1")
        assert t.id == "t1"
        assert t.name == "Task 1"
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert isinstance(t.created_at, datetime)
        assert t.result is None

    def test_create_with_custom_values(self):
        t = Task(
            id="t2",
            name="Custom",
            priority=10,
            dependencies=["t1"],
            max_retries=5,
        )
        assert t.priority == 10
        assert t.dependencies == ["t1"]
        assert t.max_retries == 5

    def test_priority_lower_bound(self):
        Task(id="ok", name="ok", priority=1)  # should not raise

    def test_priority_upper_bound(self):
        Task(id="ok", name="ok", priority=10)  # should not raise

    def test_priority_too_low(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="bad", priority=0)

    def test_priority_too_high(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="bad", priority=11)

    def test_created_at_is_utc(self):
        t = Task(id="t", name="t")
        assert t.created_at.tzinfo is not None


# ===================================================================
# Metrics dataclasses
# ===================================================================

class TestMetrics:

    def test_task_metrics_elapsed(self):
        m = TaskMetrics(task_id="x", start_time=100.0, end_time=105.5)
        assert m.elapsed == pytest.approx(5.5)

    def test_scheduler_metrics_defaults(self):
        sm = SchedulerMetrics()
        assert sm.total_time == 0.0
        assert sm.task_metrics == {}


# ===================================================================
# Scheduler construction & registration
# ===================================================================

class TestSchedulerConstruction:

    def test_default_concurrency(self):
        s = TaskScheduler()
        assert s._max_concurrency == 4

    def test_custom_concurrency(self):
        s = TaskScheduler(max_concurrency=8)
        assert s._max_concurrency == 8

    def test_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)

    def test_add_task(self):
        s = TaskScheduler()
        t = Task(id="a", name="A")
        s.add_task(t, _noop)
        assert s.get_task("a") is t

    def test_duplicate_task_id_raises(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="A2"), _noop)

    def test_get_task_missing_raises(self):
        s = TaskScheduler()
        with pytest.raises(KeyError):
            s.get_task("nonexistent")

    def test_tasks_property_returns_copy(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        copy = s.tasks
        assert "a" in copy
        # Mutating the copy does not affect the scheduler
        copy.pop("a")
        assert "a" in s.tasks


# ===================================================================
# Basic task execution
# ===================================================================

class TestBasicExecution:

    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        metrics = await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "a-done"
        assert "a" in metrics.task_metrics
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        s = TaskScheduler()
        for i in range(5):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _noop)

        metrics = await s.run()

        for i in range(5):
            t = s.get_task(f"t{i}")
            assert t.status == TaskStatus.COMPLETED
            assert t.result == f"t{i}-done"

        assert len(metrics.task_metrics) == 5

    @pytest.mark.asyncio
    async def test_task_result_is_stored(self):
        async def returns_42(task: Task) -> int:
            return 42

        s = TaskScheduler()
        s.add_task(Task(id="x", name="X"), returns_42)
        await s.run()
        assert s.get_task("x").result == 42

    @pytest.mark.asyncio
    async def test_metrics_property_after_run(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        await s.run()
        assert s.metrics is s._metrics
        assert s.metrics.total_time > 0


# ===================================================================
# Dependency resolution
# ===================================================================

class TestDependencyResolution:

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """A -> B -> C must execute in three sequential groups."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)

        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert [t.id for t in plan[0]] == ["a"]
        assert [t.id for t in plan[1]] == ["b"]
        assert [t.id for t in plan[2]] == ["c"]

        metrics = await s.run()
        for tid in ("a", "b", "c"):
            assert s.get_task(tid).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """
        Diamond:  A -> B, A -> C, B -> D, C -> D
        Groups: [A], [B, C], [D]
        """
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _noop)
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), _noop)

        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert {t.id for t in plan[0]} == {"a"}
        assert {t.id for t in plan[1]} == {"b", "c"}
        assert {t.id for t in plan[2]} == {"d"}

        await s.run()
        for tid in ("a", "b", "c", "d"):
            assert s.get_task(tid).status == TaskStatus.COMPLETED

    def test_priority_ordering_within_group(self):
        """Within the same tier, tasks are sorted by priority descending."""
        s = TaskScheduler()
        s.add_task(Task(id="low", name="Low", priority=1), _noop)
        s.add_task(Task(id="mid", name="Mid", priority=5), _noop)
        s.add_task(Task(id="high", name="High", priority=10), _noop)

        plan = s.get_execution_plan()
        # All independent -> single group, sorted high->mid->low
        assert len(plan) == 1
        ids = [t.id for t in plan[0]]
        assert ids == ["high", "mid", "low"]

    def test_unknown_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["missing"]), _noop)
        with pytest.raises(ValueError, match="unknown task 'missing'"):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_multi_tier_dependencies(self):
        """
        t1 (no deps), t2 (no deps), t3 depends on t1, t4 depends on t2,
        t5 depends on t3 and t4.
        Groups: [t1, t2], [t3, t4], [t5]
        """
        s = TaskScheduler()
        s.add_task(Task(id="t1", name="T1"), _noop)
        s.add_task(Task(id="t2", name="T2"), _noop)
        s.add_task(Task(id="t3", name="T3", dependencies=["t1"]), _noop)
        s.add_task(Task(id="t4", name="T4", dependencies=["t2"]), _noop)
        s.add_task(Task(id="t5", name="T5", dependencies=["t3", "t4"]), _noop)

        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert {t.id for t in plan[0]} == {"t1", "t2"}
        assert {t.id for t in plan[1]} == {"t3", "t4"}
        assert {t.id for t in plan[2]} == {"t5"}

        await s.run()
        for tid in ("t1", "t2", "t3", "t4", "t5"):
            assert s.get_task(tid).status == TaskStatus.COMPLETED


# ===================================================================
# Circular dependency detection
# ===================================================================

class TestCircularDependency:

    def test_self_loop(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_two_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_three_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_cycle_attribute_is_populated(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X", dependencies=["y"]), _noop)
        s.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop)
        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert exc_info.value.cycle is not None
        # The cycle should contain the offending nodes
        assert set(exc_info.value.cycle) >= {"x", "y"}

    def test_cycle_error_message_without_cycle_list(self):
        err = CircularDependencyError()
        assert "Circular dependency detected in the task graph" in str(err)
        assert err.cycle is None

    def test_cycle_error_message_with_cycle_list(self):
        err = CircularDependencyError(cycle=["a", "b", "a"])
        assert "a -> b -> a" in str(err)

    @pytest.mark.asyncio
    async def test_run_raises_on_cycle(self):
        """scheduler.run() also detects cycles (it calls get_execution_plan)."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            await s.run()


# ===================================================================
# Retry logic with exponential backoff
# ===================================================================

class TestRetryLogic:

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failure(self):
        """Task fails once, then succeeds on retry."""
        coro = _make_failing_coro(fail_times=1)
        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=3), coro)

        # Patch sleep to avoid real waiting
        original_sleep = asyncio.sleep
        sleep_durations: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)
            # Don't actually sleep in tests

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        t = s.get_task("r")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "r-recovered"
        assert t.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_fails_after_max_retries(self):
        """Task always fails -> should exhaust retries and end FAILED."""
        coro = _make_always_failing_coro()
        s = TaskScheduler()
        s.add_task(Task(id="f", name="Fail", max_retries=2), coro)

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        t = s.get_task("f")
        assert t.status == TaskStatus.FAILED
        assert isinstance(t.result, RuntimeError)
        # retry_count should exceed max_retries (max_retries=2 => 3 attempts total)
        assert t.retry_count == 3  # tried original + 2 retries, then retry_count incremented one more

    @pytest.mark.asyncio
    async def test_exponential_backoff_durations(self):
        """Verify that sleep durations follow 2^(retry-1) pattern."""
        coro = _make_failing_coro(fail_times=3)
        s = TaskScheduler()
        s.add_task(Task(id="eb", name="ExpBackoff", max_retries=5), coro)

        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        # 3 failures -> 3 backoff sleeps: 2^0=1, 2^1=2, 2^2=4
        assert sleep_durations == [1, 2, 4]

    @pytest.mark.asyncio
    async def test_retry_metrics_recorded(self):
        """TaskMetrics should reflect retry count."""
        coro = _make_failing_coro(fail_times=2)
        s = TaskScheduler()
        s.add_task(Task(id="m", name="MetricRetry", max_retries=3), coro)

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            metrics = await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert "m" in metrics.task_metrics
        assert metrics.task_metrics["m"].retries == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self):
        """With max_retries=0, the first failure is final."""
        coro = _make_always_failing_coro()
        s = TaskScheduler()
        s.add_task(Task(id="z", name="NoRetry", max_retries=0), coro)

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        t = s.get_task("z")
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1


# ===================================================================
# Concurrency control
# ===================================================================

class TestConcurrencyControl:

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At no point should more tasks run simultaneously than max_concurrency."""
        max_concurrent = 2
        active_count = 0
        peak_concurrent = 0

        async def track_concurrency(task: Task) -> str:
            nonlocal active_count, peak_concurrent
            active_count += 1
            if active_count > peak_concurrent:
                peak_concurrent = active_count
            await asyncio.sleep(0.05)
            active_count -= 1
            return f"{task.id}-done"

        s = TaskScheduler(max_concurrency=max_concurrent)
        for i in range(6):
            s.add_task(Task(id=f"c{i}", name=f"C{i}"), track_concurrency)

        await s.run()

        assert peak_concurrent <= max_concurrent
        for i in range(6):
            assert s.get_task(f"c{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrency_one_is_serial(self):
        """With max_concurrency=1, tasks run strictly one at a time."""
        execution_order: list[str] = []

        async def record_order(task: Task) -> str:
            execution_order.append(f"{task.id}-start")
            await asyncio.sleep(0.01)
            execution_order.append(f"{task.id}-end")
            return task.id

        s = TaskScheduler(max_concurrency=1)
        # Three independent tasks all in one group
        for i in range(3):
            s.add_task(Task(id=f"s{i}", name=f"S{i}"), record_order)

        await s.run()

        # With serial execution, each task must end before the next starts.
        # Verify no interleaving: the pattern should be
        # start-end-start-end-start-end
        for idx in range(0, len(execution_order), 2):
            tid = execution_order[idx].replace("-start", "")
            assert execution_order[idx] == f"{tid}-start"
            assert execution_order[idx + 1] == f"{tid}-end"

    @pytest.mark.asyncio
    async def test_all_tasks_complete_under_concurrency(self):
        """Even with a tight limit, every task eventually completes."""
        s = TaskScheduler(max_concurrency=2)
        for i in range(10):
            s.add_task(Task(id=f"w{i}", name=f"W{i}"), _slow)

        await s.run()

        for i in range(10):
            assert s.get_task(f"w{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrency_with_dependencies(self):
        """Concurrency limit applies within each dependency tier."""
        active_count = 0
        peak_concurrent = 0

        async def track(task: Task) -> str:
            nonlocal active_count, peak_concurrent
            active_count += 1
            if active_count > peak_concurrent:
                peak_concurrent = active_count
            await asyncio.sleep(0.03)
            active_count -= 1
            return task.id

        s = TaskScheduler(max_concurrency=2)
        # Tier 0: four independent tasks (should be limited to 2 at a time)
        for i in range(4):
            s.add_task(Task(id=f"tier0_{i}", name=f"T0-{i}"), track)
        # Tier 1: depends on all tier-0 tasks
        s.add_task(
            Task(id="tier1", name="T1",
                 dependencies=[f"tier0_{i}" for i in range(4)]),
            track,
        )

        await s.run()

        assert peak_concurrent <= 2
        assert s.get_task("tier1").status == TaskStatus.COMPLETED


# ===================================================================
# Observer / event system
# ===================================================================

class TestObserverEvents:

    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_start", lambda task: started.append(task.id))

        await s.run()

        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", lambda task: completed.append(task.id))

        await s.run()

        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="f", name="F", max_retries=0), _make_always_failing_coro())
        s.on("on_task_fail", lambda task, exc: failed.append(task.id))

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert "f" in failed

    @pytest.mark.asyncio
    async def test_async_callback_supported(self):
        called = False

        async def async_cb(task: Task) -> None:
            nonlocal called
            called = True

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", async_cb)

        await s.run()

        assert called is True

    @pytest.mark.asyncio
    async def test_multiple_listeners_per_event(self):
        calls: list[int] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)

        s.on("on_task_start", lambda task: calls.append(1))
        s.on("on_task_start", lambda task: calls.append(2))

        await s.run()

        assert calls == [1, 2]

    @pytest.mark.asyncio
    async def test_event_lifecycle_order(self):
        """start fires before complete for a successful task."""
        events: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)

        s.on("on_task_start", lambda task: events.append("start"))
        s.on("on_task_complete", lambda task: events.append("complete"))

        await s.run()

        assert events == ["start", "complete"]


# ===================================================================
# Failed-dependency cascading
# ===================================================================

class TestFailedDependencyCascade:

    @pytest.mark.asyncio
    async def test_downstream_fails_when_dependency_fails(self):
        """If task A fails, task B (depends on A) should be marked FAILED."""
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _noop,
        )

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED
        # B's result should indicate the dependency failure
        assert isinstance(s.get_task("b").result, RuntimeError)
        assert "Dependency tasks failed" in str(s.get_task("b").result)

    @pytest.mark.asyncio
    async def test_deep_cascade(self):
        """A -> B -> C: if A fails, both B and C should fail."""
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED
        assert s.get_task("c").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_partial_failure_in_diamond(self):
        """
        Diamond: A -> B, A -> C, B -> D, C -> D
        If B fails but C succeeds, D should still fail (B is a dependency).
        """
        call_count = 0

        async def fail_once(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if task.id == "b":
                raise RuntimeError("B fails")
            return f"{task.id}-done"

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"], max_retries=0), fail_once)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _noop)
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), _noop)

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("b").status == TaskStatus.FAILED
        assert s.get_task("c").status == TaskStatus.COMPLETED
        assert s.get_task("d").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_on_task_fail_fires_for_cascaded_failure(self):
        """The on_task_fail event should fire for tasks that fail due to
        dependency failures, not just tasks that raise exceptions."""
        failed_ids: list[str] = []
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.on("on_task_fail", lambda task, *args: failed_ids.append(task.id))

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            pass

        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        try:
            await s.run()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert "a" in failed_ids
        assert "b" in failed_ids


# ===================================================================
# Execution plan (without running)
# ===================================================================

class TestExecutionPlan:

    def test_empty_scheduler(self):
        s = TaskScheduler()
        plan = s.get_execution_plan()
        assert plan == []

    def test_single_task_plan(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert plan[0][0].id == "a"

    def test_plan_does_not_mutate_task_status(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.get_execution_plan()
        assert s.get_task("a").status == TaskStatus.PENDING


# ===================================================================
# Scheduler metrics timing
# ===================================================================

class TestSchedulerTiming:

    @pytest.mark.asyncio
    async def test_total_time_is_positive(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        metrics = await s.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_task_elapsed_is_positive(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _slow)
        metrics = await s.run()
        assert metrics.task_metrics["a"].elapsed > 0
