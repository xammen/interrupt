"""
test_task_scheduler.py

Comprehensive pytest test suite for task_scheduler.py covering:
- Basic task execution
- Dependency resolution
- Circular dependency detection
- Retry logic with exponential backoff
- Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    DuplicateTaskError,
    Task,
    TaskScheduler,
    TaskStatus,
    UnknownDependencyError,
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


def succeeding_factory(return_value=None):
    """Return a coroutine factory that always succeeds."""
    async def coro():
        return return_value
    return coro


def failing_factory(exc: Exception | None = None):
    """Return a coroutine factory that always raises."""
    if exc is None:
        exc = RuntimeError("deliberate failure")
    async def coro():
        raise exc
    return coro


def fail_n_times_factory(n: int, return_value=None):
    """Return a coroutine factory that fails the first *n* calls then succeeds."""
    calls = {"count": 0}
    async def coro():
        calls["count"] += 1
        if calls["count"] <= n:
            raise RuntimeError(f"transient failure #{calls['count']}")
        return return_value
    return coro


# ---------------------------------------------------------------------------
# Task dataclass validation
# ---------------------------------------------------------------------------


class TestTaskValidation:
    def test_valid_priority_boundaries(self):
        t1 = make_task("a", priority=1)
        t10 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_zero(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=0)

    def test_invalid_priority_eleven(self):
        with pytest.raises(ValueError):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("a")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_is_zero(self):
        t = make_task("a")
        assert t.retry_count == 0


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------


class TestTaskRegistration:
    def test_add_task_succeeds(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, succeeding_factory())
        assert "t1" in scheduler._tasks

    def test_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, succeeding_factory())
        with pytest.raises(DuplicateTaskError):
            scheduler.add_task(make_task("t1"), succeeding_factory())

    def test_unknown_dependency_raises_on_plan(self):
        scheduler = TaskScheduler()
        task = make_task("t1", dependencies=["ghost"])
        scheduler.add_task(task, succeeding_factory())
        with pytest.raises(UnknownDependencyError):
            scheduler.get_execution_plan()

    def test_unknown_dependency_raises_on_run(self):
        scheduler = TaskScheduler()
        task = make_task("t1", dependencies=["ghost"])
        scheduler.add_task(task, succeeding_factory())
        with pytest.raises(UnknownDependencyError):
            asyncio.get_event_loop().run_until_complete(scheduler.run())


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), succeeding_factory(return_value=42))
        results = await scheduler.run()
        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t1"].result == 42

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_complete(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(make_task(f"t{i}"), succeeding_factory(return_value=i))
        results = await scheduler.run()
        for i in range(5):
            assert results[f"t{i}"].status == TaskStatus.COMPLETED
            assert results[f"t{i}"].result == i

    @pytest.mark.asyncio
    async def test_failing_task_marked_failed(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(make_task("t1", max_retries=0), failing_factory())
        results = await scheduler.run()
        assert results["t1"].status == TaskStatus.FAILED
        assert isinstance(results["t1"].result, RuntimeError)

    @pytest.mark.asyncio
    async def test_task_result_stores_return_value(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), succeeding_factory(return_value="hello"))
        results = await scheduler.run()
        assert results["t1"].result == "hello"

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), succeeding_factory())
        await scheduler.run()
        assert scheduler.metrics.total_elapsed is not None
        assert scheduler.metrics.total_elapsed >= 0
        assert "t1" in scheduler.metrics.per_task
        assert scheduler.metrics.per_task["t1"].elapsed is not None


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_runs_after_dependency(self):
        execution_order: List[str] = []

        async def record(task_id: str):
            execution_order.append(task_id)

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), lambda: record("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]), lambda: record("b"))

        await scheduler.run()
        assert execution_order.index("a") < execution_order.index("b")

    @pytest.mark.asyncio
    async def test_chain_of_three_dependencies(self):
        execution_order: List[str] = []

        async def record(task_id: str):
            execution_order.append(task_id)
            return task_id

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), lambda: record("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]), lambda: record("b"))
        scheduler.add_task(make_task("c", dependencies=["b"]), lambda: record("c"))

        await scheduler.run()
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """a → b, a → c, b+c → d."""
        execution_order: List[str] = []

        async def record(task_id: str):
            execution_order.append(task_id)

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), lambda: record("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]), lambda: record("b"))
        scheduler.add_task(make_task("c", dependencies=["a"]), lambda: record("c"))
        scheduler.add_task(make_task("d", dependencies=["b", "c"]), lambda: record("d"))

        await scheduler.run()
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("a") < execution_order.index("c")
        assert execution_order.index("b") < execution_order.index("d")
        assert execution_order.index("c") < execution_order.index("d")

    def test_execution_plan_returns_correct_groups(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"), succeeding_factory())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_factory())
        scheduler.add_task(make_task("c", dependencies=["a"]), succeeding_factory())

        plan = scheduler.get_execution_plan()
        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}

    def test_execution_plan_single_task(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("solo"), succeeding_factory())
        plan = scheduler.get_execution_plan()
        assert plan == [["solo"]]

    def test_execution_plan_priority_ordering_within_group(self):
        """Within one group, higher-priority tasks come first."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("low", priority=1), succeeding_factory())
        scheduler.add_task(make_task("high", priority=9), succeeding_factory())
        scheduler.add_task(make_task("mid", priority=5), succeeding_factory())

        plan = scheduler.get_execution_plan()
        # All independent → single group
        assert len(plan) == 1
        assert plan[0] == ["high", "mid", "low"]

    @pytest.mark.asyncio
    async def test_all_tasks_complete_even_when_dependency_fails(self):
        """Dependents should still be attempted after a failed dependency."""
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(make_task("a", max_retries=0), failing_factory())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_factory())

        results = await scheduler.run()
        assert results["a"].status == TaskStatus.FAILED
        # b should still run and complete
        assert results["b"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDependencyDetection:
    def test_simple_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), succeeding_factory())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_factory())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]), succeeding_factory())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]), succeeding_factory())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_factory())
        scheduler.add_task(make_task("c", dependencies=["b"]), succeeding_factory())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_cycle_also_raises_on_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), succeeding_factory())
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_factory())
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_partial_cycle_in_larger_graph_raises(self):
        """Cycle exists among a subset; nodes outside the cycle are fine."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("root"), succeeding_factory())
        scheduler.add_task(make_task("x", dependencies=["root", "y"]), succeeding_factory())
        scheduler.add_task(make_task("y", dependencies=["x"]), succeeding_factory())
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_error_message_contains_cycle_nodes(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("alpha", dependencies=["beta"]), succeeding_factory())
        scheduler.add_task(make_task("beta", dependencies=["alpha"]), succeeding_factory())
        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()
        msg = str(exc_info.value)
        assert "alpha" in msg or "beta" in msg


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_succeeds_after_transient_failure(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(
            make_task("t1", max_retries=3),
            fail_n_times_factory(2, return_value="ok"),
        )
        results = await scheduler.run()
        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t1"].result == "ok"
        assert results["t1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(
            make_task("t1", max_retries=2),
            failing_factory(),
        )
        results = await scheduler.run()
        assert results["t1"].status == TaskStatus.FAILED
        assert results["t1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_no_retries_fails_immediately(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(
            make_task("t1", max_retries=0),
            failing_factory(),
        )
        results = await scheduler.run()
        assert results["t1"].status == TaskStatus.FAILED
        assert results["t1"].retry_count == 0

    @pytest.mark.asyncio
    async def test_retry_count_increments_correctly(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(
            make_task("t1", max_retries=3),
            fail_n_times_factory(3, return_value="done"),
        )
        results = await scheduler.run()
        assert results["t1"].retry_count == 3
        assert results["t1"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays_are_applied(self):
        """Verify asyncio.sleep is called with exponentially increasing delays."""
        sleep_calls: List[float] = []

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            # Do not actually sleep to keep tests fast
            await original_sleep(0)

        scheduler = TaskScheduler(base_retry_delay=1.0)
        scheduler.add_task(
            make_task("t1", max_retries=3),
            failing_factory(),
        )

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            await scheduler.run()

        # Should have slept for 1, 2, 4 seconds (base * 2^0, 2^1, 2^2)
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_result_stores_last_exception_on_failure(self):
        exc = ValueError("specific error")
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(make_task("t1", max_retries=0), failing_factory(exc))
        results = await scheduler.run()
        assert results["t1"].result is exc

    @pytest.mark.asyncio
    async def test_metrics_record_retry_count(self):
        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.add_task(
            make_task("t1", max_retries=2),
            fail_n_times_factory(1, return_value="ok"),
        )
        await scheduler.run()
        assert scheduler.metrics.per_task["t1"].retry_count == 1


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_max_concurrency_one_runs_serially(self):
        """With max_concurrency=1 no two tasks overlap."""
        active: List[str] = []
        overlap_detected = False

        async def tracked_coro(task_id: str):
            nonlocal overlap_detected
            active.append(task_id)
            if len(active) > 1:
                overlap_detected = True
            await asyncio.sleep(0)
            active.remove(task_id)
            return task_id

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(4):
            tid = f"t{i}"
            scheduler.add_task(make_task(tid), lambda t=tid: tracked_coro(t))

        await scheduler.run()
        assert not overlap_detected

    @pytest.mark.asyncio
    async def test_unlimited_concurrency_runs_all_in_parallel(self):
        """max_concurrency=0 means unlimited; all independent tasks start at once."""
        started: List[str] = []
        barrier = asyncio.Event()
        n = 5

        async def coro(task_id: str):
            started.append(task_id)
            await barrier.wait()
            return task_id

        scheduler = TaskScheduler(max_concurrency=0)
        for i in range(n):
            tid = f"t{i}"
            scheduler.add_task(make_task(tid), lambda t=tid: coro(t))

        run_task = asyncio.ensure_future(scheduler.run())
        # Give the event loop a chance to start all coroutines
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(started) == n, "All tasks should have started concurrently"
        barrier.set()
        await run_task

    @pytest.mark.asyncio
    async def test_concurrency_limit_two_respected(self):
        """Never more than 2 tasks active simultaneously."""
        active: set[str] = set()
        max_active_seen = 0
        lock = asyncio.Lock()

        async def slow_coro(task_id: str):
            nonlocal max_active_seen
            async with lock:
                active.add(task_id)
                if len(active) > max_active_seen:
                    max_active_seen = len(active)
            await asyncio.sleep(0.01)
            async with lock:
                active.discard(task_id)
            return task_id

        scheduler = TaskScheduler(max_concurrency=2)
        for i in range(6):
            tid = f"t{i}"
            scheduler.add_task(make_task(tid), lambda t=tid: slow_coro(t))

        await scheduler.run()
        assert max_active_seen <= 2

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_concurrency_limit(self):
        """All tasks finish regardless of concurrency cap."""
        scheduler = TaskScheduler(max_concurrency=3)
        n = 10
        for i in range(n):
            scheduler.add_task(make_task(f"t{i}"), succeeding_factory(return_value=i))

        results = await scheduler.run()
        assert len(results) == n
        for i in range(n):
            assert results[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dependent_tasks_respect_concurrency_in_each_group(self):
        """Concurrency limit applies within each topological group."""
        active: set[str] = set()
        max_seen = 0
        lock = asyncio.Lock()

        async def coro(task_id: str):
            nonlocal max_seen
            async with lock:
                active.add(task_id)
                max_seen = max(max_seen, len(active))
            await asyncio.sleep(0.01)
            async with lock:
                active.discard(task_id)

        scheduler = TaskScheduler(max_concurrency=2)
        scheduler.add_task(make_task("root"), succeeding_factory())
        for i in range(4):
            tid = f"child{i}"
            scheduler.add_task(
                make_task(tid, dependencies=["root"]),
                lambda t=tid: coro(t),
            )

        await scheduler.run()
        assert max_seen <= 2


# ---------------------------------------------------------------------------
# Observer / event callbacks
# ---------------------------------------------------------------------------


class TestObserverCallbacks:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started = []

        async def handler(task):
            started.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_start(handler)
        scheduler.add_task(make_task("t1"), succeeding_factory())
        await scheduler.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called_on_success(self):
        completed = []

        async def handler(task):
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_complete(handler)
        scheduler.add_task(make_task("t1"), succeeding_factory())
        await scheduler.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called_on_permanent_failure(self):
        failed = []

        async def handler(task):
            failed.append(task.id)

        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.on_task_fail(handler)
        scheduler.add_task(make_task("t1", max_retries=0), failing_factory())
        await scheduler.run()
        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_on_task_complete_not_called_on_failure(self):
        completed = []

        async def handler(task):
            completed.append(task.id)

        scheduler = TaskScheduler(base_retry_delay=0)
        scheduler.on_task_complete(handler)
        scheduler.add_task(make_task("t1", max_retries=0), failing_factory())
        await scheduler.run()
        assert "t1" not in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_not_called_on_success(self):
        failed = []

        async def handler(task):
            failed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_fail(handler)
        scheduler.add_task(make_task("t1"), succeeding_factory())
        await scheduler.run()
        assert "t1" not in failed

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self):
        calls = []

        async def cb1(task):
            calls.append(("cb1", task.id))

        async def cb2(task):
            calls.append(("cb2", task.id))

        scheduler = TaskScheduler()
        scheduler.on_task_complete(cb1)
        scheduler.on_task_complete(cb2)
        scheduler.add_task(make_task("t1"), succeeding_factory())
        await scheduler.run()
        assert ("cb1", "t1") in calls
        assert ("cb2", "t1") in calls
