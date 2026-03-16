"""
test_task_scheduler.py
======================
Comprehensive pytest suite for task_scheduler.py.

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
    *,
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


async def success_fn(value=None):
    """Simple coroutine that returns a value."""
    return value


async def failing_fn():
    """Coroutine that always raises."""
    raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_boundaries(self):
        t1 = make_task("t", priority=1)
        t10 = make_task("t2", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=0)
        with pytest.raises(ValueError):
            Task(id="y", name="y", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_is_zero(self):
        t = make_task("t")
        assert t.retry_count == 0

    def test_result_defaults_to_none(self):
        t = make_task("t")
        assert t.result is None


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_when_not_started(self):
        m = TaskMetrics(task_id="t")
        assert m.elapsed is None

    def test_elapsed_none_when_only_start(self):
        m = TaskMetrics(task_id="t", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed_correctly(self):
        m = TaskMetrics(task_id="t", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_elapsed_none_when_not_run(self):
        sm = SchedulerMetrics()
        assert sm.total_elapsed is None

    def test_total_elapsed_computed_correctly(self):
        sm = SchedulerMetrics(total_start=0.0, total_end=5.0)
        assert sm.total_elapsed == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_calls_registered_handler(self):
        bus = EventBus()
        received = []

        async def handler(task):
            received.append(task.id)

        t = make_task("t")
        bus.subscribe("on_task_start", handler)
        await bus.emit("on_task_start", t)
        assert received == ["t"]

    @pytest.mark.asyncio
    async def test_emit_calls_multiple_handlers(self):
        bus = EventBus()
        calls = []

        async def h1(task):
            calls.append("h1")

        async def h2(task):
            calls.append("h2")

        t = make_task("t")
        bus.subscribe("ev", h1)
        bus.subscribe("ev", h2)
        await bus.emit("ev", t)
        assert calls == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_emit_unregistered_event_is_noop(self):
        bus = EventBus()
        t = make_task("t")
        # Should not raise
        await bus.emit("nonexistent", t)

    @pytest.mark.asyncio
    async def test_handler_not_called_for_different_event(self):
        bus = EventBus()
        calls = []

        async def handler(task):
            calls.append(task.id)

        t = make_task("t")
        bus.subscribe("on_task_start", handler)
        await bus.emit("on_task_complete", t)
        assert calls == []


# ---------------------------------------------------------------------------
# TaskScheduler – registration
# ---------------------------------------------------------------------------

class TestTaskSchedulerRegistration:
    def test_add_task_succeeds(self):
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, success_fn)

    def test_duplicate_task_raises(self):
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, success_fn)
        with pytest.raises(DuplicateTaskError):
            sched.add_task(make_task("t1"), success_fn)

    def test_missing_dependency_raises_on_plan(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1", dependencies=["ghost"]), success_fn)
        with pytest.raises(MissingDependencyError):
            sched.get_execution_plan()


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_executes_and_returns_result(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), lambda: success_fn(42))
        results = await sched.run()
        assert results["t1"] == 42

    @pytest.mark.asyncio
    async def test_task_status_completed_after_run(self):
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, lambda: success_fn("ok"))
        await sched.run()
        assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        sched = TaskScheduler()
        for i in range(3):
            sched.add_task(make_task(f"t{i}"), lambda i=i: success_fn(i))
        results = await sched.run()
        assert all(results[f"t{i}"] == i for i in range(3))

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), lambda: success_fn(1))
        await sched.run()
        assert sched.metrics.total_elapsed is not None
        assert sched.metrics.total_elapsed >= 0
        assert "t1" in sched.metrics.tasks
        assert sched.metrics.tasks["t1"].elapsed is not None

    @pytest.mark.asyncio
    async def test_events_fired_on_success(self):
        sched = TaskScheduler()
        events = []

        async def handler(task):
            events.append(task.id)

        sched.events.subscribe("on_task_start", handler)
        sched.events.subscribe("on_task_complete", handler)
        sched.add_task(make_task("t1"), lambda: success_fn(1))
        await sched.run()
        assert events.count("t1") == 2  # start + complete

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_without_error(self):
        sched = TaskScheduler()
        results = await sched.run()
        assert results == {}


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        order = []
        sched = TaskScheduler()

        async def record(name):
            order.append(name)
            return name

        sched.add_task(make_task("a"), lambda: record("a"))
        sched.add_task(make_task("b", dependencies=["a"]), lambda: record("b"))
        await sched.run()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_chain_of_dependencies_ordered_correctly(self):
        order = []
        sched = TaskScheduler()

        async def record(name):
            order.append(name)

        sched.add_task(make_task("a"), lambda: record("a"))
        sched.add_task(make_task("b", dependencies=["a"]), lambda: record("b"))
        sched.add_task(make_task("c", dependencies=["b"]), lambda: record("c"))
        await sched.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_multiple_dependencies_all_run_first(self):
        order = []
        sched = TaskScheduler()

        async def record(name):
            order.append(name)

        sched.add_task(make_task("a"), lambda: record("a"))
        sched.add_task(make_task("b"), lambda: record("b"))
        sched.add_task(make_task("c", dependencies=["a", "b"]), lambda: record("c"))
        await sched.run()
        assert "c" in order
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_get_execution_plan_wave_structure(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", priority=5), success_fn)
        sched.add_task(make_task("b", priority=5), success_fn)
        sched.add_task(make_task("c", dependencies=["a", "b"], priority=5), success_fn)
        plan = sched.get_execution_plan()
        # Wave 0 must contain a and b, wave 1 must contain c
        assert set(plan[0]) == {"a", "b"}
        assert plan[1] == ["c"]

    def test_get_execution_plan_sorted_by_priority(self):
        sched = TaskScheduler()
        sched.add_task(make_task("low", priority=1), success_fn)
        sched.add_task(make_task("high", priority=9), success_fn)
        plan = sched.get_execution_plan()
        wave = plan[0]
        assert wave[0] == "high"
        assert wave[1] == "low"


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_self_loop_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["a"]), success_fn)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_two_node_cycle_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]), success_fn)
        sched.add_task(make_task("b", dependencies=["a"]), success_fn)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_three_node_cycle_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["c"]), success_fn)
        sched.add_task(make_task("b", dependencies=["a"]), success_fn)
        sched.add_task(make_task("c", dependencies=["b"]), success_fn)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_cycle_error_message_names_involved_tasks(self):
        sched = TaskScheduler()
        sched.add_task(make_task("x", dependencies=["y"]), success_fn)
        sched.add_task(make_task("y", dependencies=["x"]), success_fn)
        with pytest.raises(CircularDependencyError, match="x|y"):
            sched.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]), success_fn)
        sched.add_task(make_task("b", dependencies=["a"]), success_fn)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    def test_dag_without_cycle_does_not_raise(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), success_fn)
        sched.add_task(make_task("b", dependencies=["a"]), success_fn)
        sched.add_task(make_task("c", dependencies=["a"]), success_fn)
        sched.add_task(make_task("d", dependencies=["b", "c"]), success_fn)
        # Should not raise
        plan = sched.get_execution_plan()
        assert len(plan) > 0


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("not yet")
            return "done"

        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=3)
        sched.add_task(t, flaky)
        results = await sched.run()
        assert results["t1"] == "done"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_task_marked_failed_after_exhausting_retries(self):
        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=2)
        sched.add_task(t, failing_fn)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert t.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_count_incremented_correctly(self):
        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=2)
        sched.add_task(t, failing_fn)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert t.retry_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleep_called_with_correct_delays(self):
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        async def always_fail():
            raise RuntimeError("fail")

        sched = TaskScheduler(base_backoff=1.0)
        t = make_task("t1", max_retries=3)
        sched.add_task(t, always_fail)

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(RuntimeError):
                await sched.run()

        # Backoff for attempt 0 = 1.0 * 2^0 = 1.0
        # Backoff for attempt 1 = 1.0 * 2^1 = 2.0
        # Backoff for attempt 2 = 1.0 * 2^2 = 4.0
        # Attempt 3 (max) = no sleep (exhausted)
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_on_task_fail_event_fired_for_failed_task(self):
        failed = []

        async def on_fail(task):
            failed.append(task.id)

        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=0)
        sched.add_task(t, failing_fn)
        sched.events.subscribe("on_task_fail", on_fail)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_on_task_fail_event_not_fired_for_successful_task(self):
        failed = []

        async def on_fail(task):
            failed.append(task.id)

        sched = TaskScheduler(base_backoff=0.0)
        sched.add_task(make_task("t1"), lambda: success_fn(1))
        sched.events.subscribe("on_task_fail", on_fail)
        await sched.run()
        assert failed == []

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_on_first_error(self):
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=0)
        sched.add_task(t, fail_once)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_metrics_retry_count_matches_task_retry_count(self):
        sched = TaskScheduler(base_backoff=0.0)
        t = make_task("t1", max_retries=2)
        sched.add_task(t, failing_fn)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert sched.metrics.tasks["t1"].retry_count == t.retry_count


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_max_concurrent_one_serializes_execution(self):
        running = []
        peak = [0]

        async def track():
            running.append(1)
            peak[0] = max(peak[0], len(running))
            await asyncio.sleep(0.01)
            running.pop()
            return 1

        sched = TaskScheduler(max_concurrent=1)
        for i in range(4):
            sched.add_task(make_task(f"t{i}"), track)
        await sched.run()
        assert peak[0] == 1

    @pytest.mark.asyncio
    async def test_max_concurrent_respected_under_load(self):
        running = []
        peak = [0]
        limit = 3

        async def track():
            running.append(1)
            peak[0] = max(peak[0], len(running))
            await asyncio.sleep(0.02)
            running.pop()
            return 1

        sched = TaskScheduler(max_concurrent=limit)
        for i in range(6):
            sched.add_task(make_task(f"t{i}"), track)
        await sched.run()
        assert peak[0] <= limit

    @pytest.mark.asyncio
    async def test_all_tasks_complete_despite_concurrency_limit(self):
        sched = TaskScheduler(max_concurrent=2)
        for i in range(5):
            sched.add_task(make_task(f"t{i}"), lambda i=i: success_fn(i))
        results = await sched.run()
        assert len(results) == 5
        for i in range(5):
            assert results[f"t{i}"] == i

    @pytest.mark.asyncio
    async def test_concurrent_tasks_in_same_wave_run_in_parallel(self):
        """Tasks in the same wave should overlap when max_concurrent allows."""
        start_times = {}
        end_times = {}

        async def timed_task(name):
            start_times[name] = time.monotonic()
            await asyncio.sleep(0.05)
            end_times[name] = time.monotonic()
            return name

        sched = TaskScheduler(max_concurrent=5)
        for name in ("a", "b", "c"):
            sched.add_task(make_task(name), lambda n=name: timed_task(n))

        await sched.run()

        # All three should start before any one finishes (overlap)
        latest_start = max(start_times.values())
        earliest_end = min(end_times.values())
        assert latest_start < earliest_end, "Tasks did not run in parallel"

    @pytest.mark.asyncio
    async def test_semaphore_reused_across_waves(self):
        """Semaphore must constrain tasks across multiple dependency waves."""
        running = []
        peak = [0]
        limit = 2

        async def track(name):
            running.append(name)
            peak[0] = max(peak[0], len(running))
            await asyncio.sleep(0.01)
            running.pop()
            return name

        sched = TaskScheduler(max_concurrent=limit)
        # Wave 1: a, b, c (independent)
        for name in ("a", "b", "c"):
            sched.add_task(make_task(name), lambda n=name: track(n))
        # Wave 2: d depends on a
        sched.add_task(
            make_task("d", dependencies=["a"]), lambda: track("d")
        )
        await sched.run()
        assert peak[0] <= limit

    @pytest.mark.asyncio
    async def test_failed_task_does_not_block_other_tasks_in_wave(self):
        """A failing task should not prevent sibling tasks from completing."""
        sched = TaskScheduler(base_backoff=0.0, max_concurrent=5)
        sched.add_task(make_task("ok"), lambda: success_fn("ok"))
        sched.add_task(make_task("bad", max_retries=0), failing_fn)

        with pytest.raises((RuntimeError, ExceptionGroup)):
            await sched.run()

        # The successful task should still have completed
        assert sched._tasks["ok"].status == TaskStatus.COMPLETED
        assert sched._tasks["bad"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_multiple_failures_raise_exception_group(self):
        sched = TaskScheduler(base_backoff=0.0, max_concurrent=5)
        for i in range(3):
            sched.add_task(make_task(f"bad{i}", max_retries=0), failing_fn)

        with pytest.raises((ExceptionGroup, RuntimeError)):
            await sched.run()
