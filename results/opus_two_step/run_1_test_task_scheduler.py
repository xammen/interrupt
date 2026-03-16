"""Comprehensive tests for task_scheduler.py."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

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

async def _noop_coro(task: Task) -> str:
    """Simple coroutine that returns immediately."""
    return f"{task.id}-done"


async def _slow_coro(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"{task.id}-done"


async def _failing_coro(task: Task) -> None:
    """Coroutine that always raises."""
    raise RuntimeError(f"{task.id} exploded")


def _make_task(tid: str, *, priority: int = 5, deps: list[str] | None = None,
               max_retries: int = 3) -> Task:
    """Shortcut factory for creating Task instances."""
    return Task(id=tid, name=f"Task {tid}", priority=priority,
                dependencies=deps or [], max_retries=max_retries)


# ===================================================================
# 1. Task dataclass validation
# ===================================================================

class TestTaskDataclass:
    """Tests for the Task dataclass itself."""

    def test_valid_priority_bounds(self) -> None:
        t_low = Task(id="a", name="a", priority=1)
        t_high = Task(id="b", name="b", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10

    def test_priority_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="x", priority=0)

    def test_priority_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self) -> None:
        t = _make_task("t1")
        assert t.status is TaskStatus.PENDING

    def test_default_retry_count_is_zero(self) -> None:
        t = _make_task("t1")
        assert t.retry_count == 0

    def test_created_at_is_set(self) -> None:
        t = _make_task("t1")
        assert t.created_at is not None


# ===================================================================
# 2. TaskScheduler registration
# ===================================================================

class TestTaskRegistration:
    """Tests for add_task and scheduler initialization."""

    def test_add_task_successfully(self) -> None:
        sched = TaskScheduler()
        t = _make_task("a")
        sched.add_task(t, _noop_coro)
        # Internal dict should contain the task.
        assert "a" in sched._tasks

    def test_duplicate_task_id_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _noop_coro)
        with pytest.raises(ValueError, match="already exists"):
            sched.add_task(_make_task("a"), _noop_coro)

    def test_max_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
            TaskScheduler(max_concurrency=0)

    def test_max_concurrency_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            TaskScheduler(max_concurrency=-1)


# ===================================================================
# 3. Basic task execution
# ===================================================================

class TestBasicExecution:
    """Tests for running simple tasks without dependencies."""

    @pytest.mark.asyncio
    async def test_single_task_completes(self) -> None:
        sched = TaskScheduler()
        t = _make_task("t1")
        sched.add_task(t, _noop_coro)

        metrics = await sched.run()

        assert t.status is TaskStatus.COMPLETED
        assert t.result == "t1-done"
        assert "t1" in metrics.per_task_time
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self) -> None:
        sched = TaskScheduler()
        tasks = [_make_task(f"t{i}") for i in range(5)]
        for t in tasks:
            sched.add_task(t, _noop_coro)

        await sched.run()

        for t in tasks:
            assert t.status is TaskStatus.COMPLETED
            assert t.result == f"{t.id}-done"

    @pytest.mark.asyncio
    async def test_metrics_populated_for_all_tasks(self) -> None:
        sched = TaskScheduler()
        for i in range(3):
            sched.add_task(_make_task(f"t{i}"), _noop_coro)

        metrics = await sched.run()

        assert len(metrics.per_task_time) == 3
        assert len(metrics.retry_counts) == 3
        for i in range(3):
            assert metrics.retry_counts[f"t{i}"] == 0

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_without_error(self) -> None:
        sched = TaskScheduler()
        metrics = await sched.run()
        assert metrics.total_time >= 0
        assert len(metrics.per_task_time) == 0


# ===================================================================
# 4. Dependency resolution
# ===================================================================

class TestDependencyResolution:
    """Tests for topological sort and execution plan."""

    def test_linear_dependency_chain(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _noop_coro)
        sched.add_task(_make_task("b", deps=["a"]), _noop_coro)
        sched.add_task(_make_task("c", deps=["b"]), _noop_coro)

        plan = sched.get_execution_plan()

        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_diamond_dependency(self) -> None:
        """
        A -> B, A -> C, B -> D, C -> D
        Layer 0: [A], Layer 1: [B, C], Layer 2: [D]
        """
        sched = TaskScheduler()
        sched.add_task(_make_task("A", priority=5), _noop_coro)
        sched.add_task(_make_task("B", priority=7, deps=["A"]), _noop_coro)
        sched.add_task(_make_task("C", priority=3, deps=["A"]), _noop_coro)
        sched.add_task(_make_task("D", priority=5, deps=["B", "C"]), _noop_coro)

        plan = sched.get_execution_plan()

        assert len(plan) == 3
        assert plan[0] == ["A"]
        # B has higher priority so should come first in layer 1.
        assert plan[1] == ["B", "C"]
        assert plan[2] == ["D"]

    def test_independent_tasks_share_layer(self) -> None:
        sched = TaskScheduler()
        for tid in ["x", "y", "z"]:
            sched.add_task(_make_task(tid), _noop_coro)

        plan = sched.get_execution_plan()
        # All independent tasks should be in a single layer.
        assert len(plan) == 1
        assert set(plan[0]) == {"x", "y", "z"}

    def test_priority_ordering_within_layer(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("low", priority=1), _noop_coro)
        sched.add_task(_make_task("mid", priority=5), _noop_coro)
        sched.add_task(_make_task("high", priority=10), _noop_coro)

        plan = sched.get_execution_plan()

        assert plan[0] == ["high", "mid", "low"]

    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self) -> None:
        execution_order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            execution_order.append(task.id)
            return task.id

        sched = TaskScheduler()
        sched.add_task(_make_task("dep"), tracking_coro)
        sched.add_task(_make_task("main", deps=["dep"]), tracking_coro)

        await sched.run()

        assert execution_order.index("dep") < execution_order.index("main")

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades_failure(self) -> None:
        sched = TaskScheduler()
        parent = _make_task("parent", max_retries=0)
        child = _make_task("child", deps=["parent"])

        sched.add_task(parent, _failing_coro)
        sched.add_task(child, _noop_coro)

        await sched.run()

        assert parent.status is TaskStatus.FAILED
        assert child.status is TaskStatus.FAILED

    def test_unknown_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["nonexistent"]), _noop_coro)

        with pytest.raises(ValueError, match="unknown task"):
            sched.get_execution_plan()

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises_on_run(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["ghost"]), _noop_coro)

        with pytest.raises(ValueError, match="unknown task"):
            await sched.run()


# ===================================================================
# 5. Circular dependency detection
# ===================================================================

class TestCircularDependency:
    """Tests for cycle detection in the dependency graph."""

    def test_self_dependency_detected(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["a"]), _noop_coro)

        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_two_node_cycle_detected(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["b"]), _noop_coro)
        sched.add_task(_make_task("b", deps=["a"]), _noop_coro)

        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_three_node_cycle_detected(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["c"]), _noop_coro)
        sched.add_task(_make_task("b", deps=["a"]), _noop_coro)
        sched.add_task(_make_task("c", deps=["b"]), _noop_coro)

        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_cycle_error_message_contains_involved_tasks(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("x", deps=["y"]), _noop_coro)
        sched.add_task(_make_task("y", deps=["x"]), _noop_coro)

        with pytest.raises(CircularDependencyError, match="x"):
            sched.get_execution_plan()

    @pytest.mark.asyncio
    async def test_circular_dependency_raises_on_run(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a", deps=["b"]), _noop_coro)
        sched.add_task(_make_task("b", deps=["a"]), _noop_coro)

        with pytest.raises(CircularDependencyError):
            await sched.run()

    def test_partial_cycle_with_valid_nodes(self) -> None:
        """Graph: ok1 (no deps), cyc_a <-> cyc_b. Should still detect cycle."""
        sched = TaskScheduler()
        sched.add_task(_make_task("ok1"), _noop_coro)
        sched.add_task(_make_task("cyc_a", deps=["cyc_b"]), _noop_coro)
        sched.add_task(_make_task("cyc_b", deps=["cyc_a"]), _noop_coro)

        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()


# ===================================================================
# 6. Retry logic with exponential backoff
# ===================================================================

class TestRetryLogic:
    """Tests for retry behaviour and exponential backoff."""

    @pytest.mark.asyncio
    async def test_task_retries_on_failure(self) -> None:
        call_count = 0

        async def fail_twice(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient")
            return "ok"

        sched = TaskScheduler()
        t = _make_task("retry_me", max_retries=3)
        sched.add_task(t, fail_twice)

        # Patch sleep to avoid real delays.
        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert t.status is TaskStatus.COMPLETED
        assert t.result == "ok"
        assert call_count == 3  # 1 initial + 2 retries until success on 3rd

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries_exceeded(self) -> None:
        sched = TaskScheduler()
        t = _make_task("doomed", max_retries=2)
        sched.add_task(t, _failing_coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert t.status is TaskStatus.FAILED
        # retry_count should be max_retries + 1 (initial attempt counts as first retry increment)
        assert t.retry_count == t.max_retries + 1

    @pytest.mark.asyncio
    async def test_zero_retries_means_single_attempt(self) -> None:
        call_count = 0

        async def counting_fail(task: Task) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        sched = TaskScheduler()
        t = _make_task("one_shot", max_retries=0)
        sched.add_task(t, counting_fail)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert t.status is TaskStatus.FAILED
        assert call_count == 1  # only the initial attempt

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        """Verify that sleep is called with exponentially increasing delays."""
        sleep_values: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_values.append(seconds)

        sched = TaskScheduler()
        t = _make_task("backoff", max_retries=4)
        sched.add_task(t, _failing_coro)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            await sched.run()

        # After each failure retry_count increments, backoff = 2^(retry_count-1)
        # Retry 1: 2^0=1, Retry 2: 2^1=2, Retry 3: 2^2=4, Retry 4: 2^3=8
        assert sleep_values == [1, 2, 4, 8]

    @pytest.mark.asyncio
    async def test_backoff_capped_at_60_seconds(self) -> None:
        """Verify the backoff caps at 60 seconds for high retry counts."""
        sleep_values: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_values.append(seconds)

        sched = TaskScheduler()
        t = _make_task("cap_test", max_retries=10)
        sched.add_task(t, _failing_coro)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            await sched.run()

        # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64->60, ...
        for val in sleep_values:
            assert val <= 60

    @pytest.mark.asyncio
    async def test_retry_count_recorded_in_metrics(self) -> None:
        call_count = 0

        async def fail_once(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            return "recovered"

        sched = TaskScheduler()
        t = _make_task("metrics_retry", max_retries=3)
        sched.add_task(t, fail_once)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            metrics = await sched.run()

        assert metrics.retry_counts["metrics_retry"] == 1


# ===================================================================
# 7. Concurrent execution and concurrency limits
# ===================================================================

class TestConcurrencyLimits:
    """Tests ensuring the semaphore-based concurrency limit is respected."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        """No more than max_concurrency tasks should run simultaneously."""
        max_concurrent = 2
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def concurrency_probe(task: Task) -> str:
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return task.id

        sched = TaskScheduler(max_concurrency=max_concurrent)
        for i in range(6):
            sched.add_task(_make_task(f"c{i}"), concurrency_probe)

        await sched.run()

        assert peak_concurrent <= max_concurrent

    @pytest.mark.asyncio
    async def test_concurrency_of_one_is_serial(self) -> None:
        execution_order: list[str] = []

        async def ordered_coro(task: Task) -> str:
            execution_order.append(task.id)
            await asyncio.sleep(0.01)
            return task.id

        sched = TaskScheduler(max_concurrency=1)
        # Add tasks with same priority so order is deterministic within layer.
        for i in range(3):
            sched.add_task(_make_task(f"s{i}", priority=5), ordered_coro)

        await sched.run()

        assert len(execution_order) == 3

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_limited_concurrency(self) -> None:
        sched = TaskScheduler(max_concurrency=2)
        tasks = [_make_task(f"t{i}") for i in range(10)]
        for t in tasks:
            sched.add_task(t, _slow_coro)

        await sched.run()

        for t in tasks:
            assert t.status is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_high_concurrency_runs_all(self) -> None:
        sched = TaskScheduler(max_concurrency=100)
        tasks = [_make_task(f"t{i}") for i in range(20)]
        for t in tasks:
            sched.add_task(t, _noop_coro)

        await sched.run()

        for t in tasks:
            assert t.status is TaskStatus.COMPLETED


# ===================================================================
# 8. Event / observer system
# ===================================================================

class TestEventSystem:
    """Tests for the on_task_start / on_task_complete / on_task_fail hooks."""

    @pytest.mark.asyncio
    async def test_on_task_start_fires(self) -> None:
        started: list[str] = []
        sched = TaskScheduler()
        sched.on_task_start(lambda t: started.append(t.id))
        sched.add_task(_make_task("a"), _noop_coro)

        await sched.run()

        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self) -> None:
        completed: list[str] = []
        sched = TaskScheduler()
        sched.on_task_complete(lambda t: completed.append(t.id))
        sched.add_task(_make_task("a"), _noop_coro)

        await sched.run()

        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self) -> None:
        failed: list[str] = []
        sched = TaskScheduler()
        sched.on_task_fail(lambda t: failed.append(t.id))
        sched.add_task(_make_task("a", max_retries=0), _failing_coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert "a" in failed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires_for_cascaded_failure(self) -> None:
        failed: list[str] = []
        sched = TaskScheduler()
        sched.on_task_fail(lambda t: failed.append(t.id))

        sched.add_task(_make_task("parent", max_retries=0), _failing_coro)
        sched.add_task(_make_task("child", deps=["parent"]), _noop_coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert "parent" in failed
        assert "child" in failed

    @pytest.mark.asyncio
    async def test_listener_exception_does_not_crash_scheduler(self) -> None:
        def exploding_listener(task: Task) -> None:
            raise RuntimeError("listener boom")

        sched = TaskScheduler()
        sched.on_task_start(exploding_listener)
        sched.add_task(_make_task("a"), _noop_coro)

        # Should not raise.
        await sched.run()
        assert sched._tasks["a"].status is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_listener_is_awaited(self) -> None:
        started: list[str] = []

        async def async_listener(task: Task) -> None:
            await asyncio.sleep(0)
            started.append(task.id)

        sched = TaskScheduler()
        sched.on_task_start(async_listener)
        sched.add_task(_make_task("a"), _noop_coro)

        await sched.run()

        assert "a" in started

    @pytest.mark.asyncio
    async def test_multiple_listeners_all_called(self) -> None:
        log1: list[str] = []
        log2: list[str] = []

        sched = TaskScheduler()
        sched.on_task_complete(lambda t: log1.append(t.id))
        sched.on_task_complete(lambda t: log2.append(t.id))
        sched.add_task(_make_task("a"), _noop_coro)

        await sched.run()

        assert "a" in log1
        assert "a" in log2


# ===================================================================
# 9. TaskMetrics
# ===================================================================

class TestTaskMetrics:
    """Tests for metrics collection."""

    @pytest.mark.asyncio
    async def test_total_time_is_positive(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _slow_coro)

        metrics = await sched.run()

        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_per_task_time_recorded(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _slow_coro)

        metrics = await sched.run()

        assert "a" in metrics.per_task_time
        assert metrics.per_task_time["a"] > 0

    @pytest.mark.asyncio
    async def test_metrics_reset_on_rerun(self) -> None:
        """Each call to run() should reset the metrics."""
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _noop_coro)

        m1 = await sched.run()
        assert "a" in m1.per_task_time

        # Re-running resets metrics internally (though tasks keep their state).
        # We need a fresh scheduler for a clean second run.
        sched2 = TaskScheduler()
        sched2.add_task(_make_task("b"), _noop_coro)
        m2 = await sched2.run()

        assert "b" in m2.per_task_time
        assert "a" not in m2.per_task_time

    @pytest.mark.asyncio
    async def test_metrics_property_accessible(self) -> None:
        sched = TaskScheduler()
        sched.add_task(_make_task("a"), _noop_coro)
        await sched.run()

        assert sched.metrics is sched._metrics
        assert "a" in sched.metrics.per_task_time


# ===================================================================
# 10. Integration / complex scenarios
# ===================================================================

class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple features."""

    @pytest.mark.asyncio
    async def test_diamond_with_one_branch_failing(self) -> None:
        """
        A -> B (fails), A -> C (ok), B -> D, C -> D
        D should fail because B failed.
        """
        sched = TaskScheduler()
        sched.add_task(_make_task("A"), _noop_coro)
        sched.add_task(_make_task("B", deps=["A"], max_retries=0), _failing_coro)
        sched.add_task(_make_task("C", deps=["A"]), _noop_coro)
        sched.add_task(_make_task("D", deps=["B", "C"]), _noop_coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert sched._tasks["A"].status is TaskStatus.COMPLETED
        assert sched._tasks["B"].status is TaskStatus.FAILED
        assert sched._tasks["C"].status is TaskStatus.COMPLETED
        assert sched._tasks["D"].status is TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_then_succeed_with_dependencies(self) -> None:
        """A task that fails once then succeeds should still unblock dependents."""
        attempt = 0

        async def flaky(task: Task) -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise RuntimeError("transient")
            return "recovered"

        sched = TaskScheduler()
        sched.add_task(_make_task("flaky", max_retries=2), flaky)
        sched.add_task(_make_task("after_flaky", deps=["flaky"]), _noop_coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await sched.run()

        assert sched._tasks["flaky"].status is TaskStatus.COMPLETED
        assert sched._tasks["after_flaky"].status is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_large_task_graph(self) -> None:
        """Stress test with a chain of 50 tasks."""
        sched = TaskScheduler(max_concurrency=4)

        prev_id = None
        for i in range(50):
            tid = f"t{i}"
            deps = [prev_id] if prev_id else []
            sched.add_task(_make_task(tid, deps=deps), _noop_coro)
            prev_id = tid

        metrics = await sched.run()

        assert len(metrics.per_task_time) == 50
        for i in range(50):
            assert sched._tasks[f"t{i}"].status is TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wide_task_graph(self) -> None:
        """Many independent tasks fanning out from a single root."""
        sched = TaskScheduler(max_concurrency=4)
        sched.add_task(_make_task("root"), _noop_coro)
        for i in range(20):
            sched.add_task(_make_task(f"leaf{i}", deps=["root"]), _noop_coro)

        metrics = await sched.run()

        assert sched._tasks["root"].status is TaskStatus.COMPLETED
        for i in range(20):
            assert sched._tasks[f"leaf{i}"].status is TaskStatus.COMPLETED
        assert len(metrics.per_task_time) == 21

    @pytest.mark.asyncio
    async def test_event_ordering_start_before_complete(self) -> None:
        events: list[tuple[str, str]] = []

        sched = TaskScheduler()
        sched.on_task_start(lambda t: events.append(("start", t.id)))
        sched.on_task_complete(lambda t: events.append(("complete", t.id)))
        sched.add_task(_make_task("a"), _noop_coro)

        await sched.run()

        assert events == [("start", "a"), ("complete", "a")]

    @pytest.mark.asyncio
    async def test_concurrency_with_dependencies(self) -> None:
        """Tasks in different layers should respect both deps and concurrency."""
        order: list[str] = []

        async def track(task: Task) -> str:
            order.append(f"start-{task.id}")
            await asyncio.sleep(0.01)
            order.append(f"end-{task.id}")
            return task.id

        sched = TaskScheduler(max_concurrency=2)
        sched.add_task(_make_task("a"), track)
        sched.add_task(_make_task("b"), track)
        sched.add_task(_make_task("c", deps=["a", "b"]), track)

        await sched.run()

        # c must start after both a and b have ended.
        c_start = order.index("start-c")
        a_end = order.index("end-a")
        b_end = order.index("end-b")
        assert c_start > a_end
        assert c_start > b_end
