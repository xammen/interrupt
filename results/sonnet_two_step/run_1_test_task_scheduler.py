"""
Comprehensive pytest test suite for task_scheduler.py.

Covers:
- Basic task execution
- Dependency resolution
- Circular dependency detection
- Retry logic with exponential backoff
- Concurrent execution respecting concurrency limits
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


def noop_factory(return_value=None):
    """Returns a coroutine factory that simply returns *return_value*."""

    async def _coro():
        return return_value

    return _coro


def failing_factory(fail_times: int, return_value=None):
    """Returns a factory whose coroutine raises ValueError the first
    *fail_times* invocations then succeeds."""
    call_count = 0

    async def _coro():
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise ValueError(f"Intentional failure #{call_count}")
        return return_value

    return _coro


def always_failing_factory():
    """Returns a factory whose coroutine always raises."""

    async def _coro():
        raise RuntimeError("Always fails")

    return _coro


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


class TestTaskDataclass:
    def test_valid_priority_bounds(self):
        t = make_task("t1", priority=1)
        assert t.priority == 1
        t = make_task("t2", priority=10)
        assert t.priority == 10

    def test_invalid_priority_below_1(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=0)

    def test_invalid_priority_above_10(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t1")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_is_zero(self):
        t = make_task("t1")
        assert t.retry_count == 0

    def test_default_result_is_none(self):
        t = make_task("t1")
        assert t.result is None


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------


class TestTaskMetrics:
    def test_elapsed_none_when_times_missing(self):
        m = TaskMetrics(task_id="t1")
        assert m.elapsed is None

    def test_elapsed_none_when_end_time_missing(self):
        m = TaskMetrics(task_id="t1", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed_correctly(self):
        m = TaskMetrics(task_id="t1", start_time=1.0, end_time=4.5)
        assert m.elapsed == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# TaskScheduler construction
# ---------------------------------------------------------------------------


class TestSchedulerConstruction:
    def test_default_max_concurrency(self):
        s = TaskScheduler()
        assert s._max_concurrency == 4

    def test_custom_max_concurrency(self):
        s = TaskScheduler(max_concurrency=2)
        assert s._max_concurrency == 2

    def test_invalid_max_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            TaskScheduler(max_concurrency=0)

    def test_total_elapsed_none_before_run(self):
        s = TaskScheduler()
        assert s.total_elapsed is None


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_runs_and_returns_result(self):
        s = TaskScheduler()
        t = make_task("t1")
        s.add_task(t, noop_factory(return_value=42))
        results = await s.run()
        assert results["t1"] == 42

    @pytest.mark.asyncio
    async def test_task_status_completed_after_run(self):
        s = TaskScheduler()
        t = make_task("t1")
        s.add_task(t, noop_factory())
        await s.run()
        assert s._tasks["t1"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        s = TaskScheduler()
        for i in range(5):
            s.add_task(make_task(f"t{i}"), noop_factory(return_value=i))
        results = await s.run()
        for i in range(5):
            assert results[f"t{i}"] == i
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_total_elapsed_set_after_run(self):
        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())
        await s.run()
        assert s.total_elapsed is not None
        assert s.total_elapsed >= 0

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())
        await s.run()
        m = s.metrics["t1"]
        assert m.start_time is not None
        assert m.end_time is not None
        assert m.elapsed is not None and m.elapsed >= 0

    @pytest.mark.asyncio
    async def test_result_stored_on_task_object(self):
        s = TaskScheduler()
        t = make_task("t1")
        s.add_task(t, noop_factory(return_value="hello"))
        await s.run()
        assert s._tasks["t1"].result == "hello"

    @pytest.mark.asyncio
    async def test_empty_scheduler_returns_empty_dict(self):
        s = TaskScheduler()
        results = await s.run()
        assert results == {}


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        execution_order: List[str] = []

        async def make_recording_coro(tid: str):
            execution_order.append(tid)

        s = TaskScheduler()
        s.add_task(make_task("a"), lambda: make_recording_coro("a"))
        s.add_task(
            make_task("b", dependencies=["a"]),
            lambda: make_recording_coro("b"),
        )
        await s.run()
        assert execution_order.index("a") < execution_order.index("b")

    @pytest.mark.asyncio
    async def test_chain_dependency_ordering(self):
        order: List[str] = []

        def recorder(tid):
            async def _c():
                order.append(tid)

            return _c

        s = TaskScheduler()
        s.add_task(make_task("a"), recorder("a"))
        s.add_task(make_task("b", dependencies=["a"]), recorder("b"))
        s.add_task(make_task("c", dependencies=["b"]), recorder("c"))
        await s.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_fan_in_dependency(self):
        """Task C depends on both A and B; A and B run concurrently first."""
        order: List[str] = []

        def recorder(tid):
            async def _c():
                order.append(tid)

            return _c

        s = TaskScheduler()
        s.add_task(make_task("a"), recorder("a"))
        s.add_task(make_task("b"), recorder("b"))
        s.add_task(make_task("c", dependencies=["a", "b"]), recorder("c"))
        await s.run()
        assert "c" in order
        assert order.index("c") > order.index("a")
        assert order.index("c") > order.index("b")

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises_task_not_found(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["nonexistent"]), noop_factory())
        with pytest.raises(TaskNotFoundError, match="nonexistent"):
            await s.run()

    def test_get_execution_plan_linear(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop_factory())
        s.add_task(make_task("b", dependencies=["a"]), noop_factory())
        plan = s.get_execution_plan()
        # a must appear in an earlier group than b
        a_level = next(i for i, g in enumerate(plan) if "a" in g)
        b_level = next(i for i, g in enumerate(plan) if "b" in g)
        assert a_level < b_level

    def test_get_execution_plan_independent_tasks_same_group(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop_factory())
        s.add_task(make_task("b"), noop_factory())
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_execution_plan_unknown_dep_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["ghost"]), noop_factory())
        with pytest.raises(TaskNotFoundError):
            s.get_execution_plan()


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDependencyDetection:
    def test_direct_cycle_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), noop_factory())
        s.add_task(make_task("b", dependencies=["a"]), noop_factory())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_three_node_cycle_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["c"]), noop_factory())
        s.add_task(make_task("b", dependencies=["a"]), noop_factory())
        s.add_task(make_task("c", dependencies=["b"]), noop_factory())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_self_dependency_raises(self):
        s = TaskScheduler()
        # A task depending on itself is a cycle of length 1
        s.add_task(make_task("a", dependencies=["a"]), noop_factory())
        with pytest.raises((CircularDependencyError, TaskNotFoundError)):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), noop_factory())
        s.add_task(make_task("b", dependencies=["a"]), noop_factory())
        with pytest.raises(CircularDependencyError):
            await s.run()

    def test_no_false_positive_for_diamond(self):
        """Diamond graph (a->b, a->c, b->d, c->d) is acyclic."""
        s = TaskScheduler()
        s.add_task(make_task("a"), noop_factory())
        s.add_task(make_task("b", dependencies=["a"]), noop_factory())
        s.add_task(make_task("c", dependencies=["a"]), noop_factory())
        s.add_task(make_task("d", dependencies=["b", "c"]), noop_factory())
        plan = s.get_execution_plan()
        assert plan  # should not raise


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_and_eventually_succeeds(self):
        s = TaskScheduler()
        t = make_task("t1", max_retries=3)
        s.add_task(t, failing_factory(fail_times=2, return_value="ok"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            results = await s.run()

        assert results["t1"] == "ok"
        assert s._tasks["t1"].status == TaskStatus.COMPLETED
        # Should have slept twice (after failure 1 and failure 2)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_task_marked_failed_when_retries_exhausted(self):
        s = TaskScheduler()
        t = make_task("t1", max_retries=2)
        s.add_task(t, always_failing_factory())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert s._tasks["t1"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_count_tracked_on_task(self):
        s = TaskScheduler()
        t = make_task("t1", max_retries=3)
        s.add_task(t, failing_factory(fail_times=2))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert s._tasks["t1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_retry_count_on_metrics(self):
        s = TaskScheduler()
        t = make_task("t1", max_retries=3)
        s.add_task(t, failing_factory(fail_times=1))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert s.metrics["t1"].retry_count >= 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay_values(self):
        """Sleep delays should follow 2**attempt pattern (capped at 60)."""
        s = TaskScheduler()
        t = make_task("t1", max_retries=3)
        s.add_task(t, failing_factory(fail_times=3, return_value=None))

        sleep_calls: List[float] = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await s.run()

        # Attempts 1, 2, 3 -> delays 2**1=2, 2**2=4, 2**3=8
        assert sleep_calls == [2, 4, 8]

    @pytest.mark.asyncio
    async def test_backoff_capped_at_60_seconds(self):
        """Ensure the delay never exceeds 60 s regardless of attempt number."""
        s = TaskScheduler()
        # max_retries=10 gives attempt values up to 11
        t = make_task("t1", max_retries=10)
        s.add_task(t, always_failing_factory())

        sleep_calls: List[float] = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await s.run()

        assert all(d <= 60 for d in sleep_calls)

    @pytest.mark.asyncio
    async def test_no_sleep_on_first_success(self):
        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await s.run()

        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_is_none_for_failed_task(self):
        s = TaskScheduler()
        t = make_task("t1", max_retries=0)
        s.add_task(t, always_failing_factory())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            results = await s.run()

        assert results["t1"] is None


# ---------------------------------------------------------------------------
# Observer pattern / event callbacks
# ---------------------------------------------------------------------------


class TestObserverPattern:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started = []

        async def on_start(task):
            started.append(task.id)

        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())
        s.on_task_start(on_start)
        await s.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called_on_success(self):
        completed = []

        async def on_complete(task):
            completed.append(task.id)

        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())
        s.on_task_complete(on_complete)
        await s.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called_on_exhausted_retries(self):
        failed = []

        async def on_fail(task):
            failed.append(task.id)

        s = TaskScheduler()
        t = make_task("t1", max_retries=1)
        s.add_task(t, always_failing_factory())
        s.on_task_fail(on_fail)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_on_task_complete_not_called_on_failure(self):
        completed = []

        async def on_complete(task):
            completed.append(task.id)

        s = TaskScheduler()
        t = make_task("t1", max_retries=0)
        s.add_task(t, always_failing_factory())
        s.on_task_complete(on_complete)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert "t1" not in completed

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_called(self):
        calls: List[str] = []

        async def cb1(task):
            calls.append("cb1")

        async def cb2(task):
            calls.append("cb2")

        s = TaskScheduler()
        s.add_task(make_task("t1"), noop_factory())
        s.on_task_complete(cb1)
        s.on_task_complete(cb2)
        await s.run()
        assert "cb1" in calls
        assert "cb2" in calls


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_max_concurrency_1_serialises_execution(self):
        """With max_concurrency=1, tasks within a level cannot overlap."""
        active: List[int] = []
        peak: List[int] = []
        running = [0]

        async def tracked_coro():
            running[0] += 1
            peak.append(running[0])
            await asyncio.sleep(0)  # yield to event loop
            running[0] -= 1

        s = TaskScheduler(max_concurrency=1)
        for i in range(4):
            s.add_task(make_task(f"t{i}"), tracked_coro)

        await s.run()
        assert max(peak) == 1

    @pytest.mark.asyncio
    async def test_max_concurrency_2_allows_two_parallel(self):
        running = [0]
        peak = [0]

        async def tracked_coro():
            running[0] += 1
            if running[0] > peak[0]:
                peak[0] = running[0]
            await asyncio.sleep(0.01)
            running[0] -= 1

        s = TaskScheduler(max_concurrency=2)
        for i in range(4):
            s.add_task(make_task(f"t{i}"), tracked_coro)

        await s.run()
        assert peak[0] <= 2

    @pytest.mark.asyncio
    async def test_concurrency_limit_never_exceeded(self):
        """Peak concurrency must never exceed the configured limit."""
        limit = 3
        running = [0]
        violations: List[int] = []

        async def tracked_coro():
            running[0] += 1
            if running[0] > limit:
                violations.append(running[0])
            await asyncio.sleep(0.005)
            running[0] -= 1

        s = TaskScheduler(max_concurrency=limit)
        for i in range(10):
            s.add_task(make_task(f"t{i}"), tracked_coro)

        await s.run()
        assert violations == [], f"Concurrency limit exceeded: {violations}"

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_strict_limit(self):
        s = TaskScheduler(max_concurrency=1)
        for i in range(6):
            s.add_task(make_task(f"t{i}"), noop_factory(return_value=i))
        results = await s.run()
        assert len(results) == 6
        for i in range(6):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_priority_ordering_within_group(self):
        """Higher-priority tasks should start before lower-priority ones."""
        order: List[str] = []

        def make_recorder(tid: str):
            async def _c():
                order.append(tid)

            return _c

        s = TaskScheduler(max_concurrency=1)
        # Add in reverse priority order
        s.add_task(make_task("low", priority=1), make_recorder("low"))
        s.add_task(make_task("mid", priority=5), make_recorder("mid"))
        s.add_task(make_task("high", priority=10), make_recorder("high"))
        await s.run()
        assert order.index("high") < order.index("mid")
        assert order.index("mid") < order.index("low")

    @pytest.mark.asyncio
    async def test_dependency_tasks_respect_concurrency(self):
        """Even in dependent chains, the semaphore must be respected."""
        limit = 2
        running = [0]
        violations: List[int] = []

        async def tracked_coro():
            running[0] += 1
            if running[0] > limit:
                violations.append(running[0])
            await asyncio.sleep(0.005)
            running[0] -= 1

        s = TaskScheduler(max_concurrency=limit)
        s.add_task(make_task("a"), tracked_coro)
        s.add_task(make_task("b"), tracked_coro)
        s.add_task(make_task("c", dependencies=["a"]), tracked_coro)
        s.add_task(make_task("d", dependencies=["b"]), tracked_coro)
        await s.run()
        assert violations == []
