"""Comprehensive tests for task_scheduler.py."""

from __future__ import annotations

import asyncio
import time
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


# =====================================================================
# Helpers
# =====================================================================

def _make_task(tid: str, *, priority: int = 5, deps: list[str] | None = None,
               max_retries: int = 3) -> Task:
    """Shorthand for creating a Task."""
    return Task(id=tid, name=f"task-{tid}", priority=priority,
                dependencies=deps or [], max_retries=max_retries)


async def _noop(task: Task) -> str:
    """Async callable that does nothing and returns a string."""
    return f"result-{task.id}"


async def _sleep_short(task: Task) -> str:
    """Async callable that sleeps briefly to simulate real work."""
    await asyncio.sleep(0.01)
    return f"result-{task.id}"


# =====================================================================
# 1. Basic task creation and validation
# =====================================================================

class TestTaskCreation:
    """Tests for the Task dataclass and its validation."""

    def test_create_task_defaults(self):
        t = Task(id="a", name="alpha", priority=5)
        assert t.id == "a"
        assert t.name == "alpha"
        assert t.priority == 5
        assert t.status == TaskStatus.PENDING
        assert t.dependencies == []
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_create_task_with_dependencies(self):
        t = Task(id="b", name="beta", priority=3, dependencies=["a"])
        assert t.dependencies == ["a"]

    @pytest.mark.parametrize("bad_priority", [0, -1, 11, 100])
    def test_priority_out_of_range_raises(self, bad_priority: int):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="x", name="x", priority=bad_priority)

    @pytest.mark.parametrize("good_priority", [1, 5, 10])
    def test_priority_boundary_values(self, good_priority: int):
        t = Task(id="x", name="x", priority=good_priority)
        assert t.priority == good_priority


class TestTaskMetrics:
    """Tests for TaskMetrics and SchedulerMetrics dataclasses."""

    def test_scheduler_metrics_total_retries(self):
        sm = SchedulerMetrics()
        sm.task_metrics["a"] = TaskMetrics(task_id="a", retry_count=2)
        sm.task_metrics["b"] = TaskMetrics(task_id="b", retry_count=1)
        assert sm.total_retries == 3

    def test_scheduler_metrics_total_retries_empty(self):
        sm = SchedulerMetrics()
        assert sm.total_retries == 0


# =====================================================================
# 2. Task registration (add / remove)
# =====================================================================

class TestTaskRegistration:
    """Tests for add_task and remove_task."""

    def test_add_task(self):
        scheduler = TaskScheduler()
        t = _make_task("a")
        scheduler.add_task(t, _noop)
        assert "a" in scheduler.tasks
        assert scheduler.get_task("a") is t

    def test_add_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(_make_task("a"), _noop)

    def test_remove_pending_task(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        scheduler.remove_task("a")
        assert "a" not in scheduler.tasks

    def test_remove_unknown_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(KeyError, match="Unknown task id"):
            scheduler.remove_task("nonexistent")

    @pytest.mark.asyncio
    async def test_remove_non_pending_task_raises(self):
        scheduler = TaskScheduler()
        t = _make_task("a")
        scheduler.add_task(t, _noop)
        await scheduler.run()
        with pytest.raises(RuntimeError, match="Cannot remove task"):
            scheduler.remove_task("a")

    def test_get_unknown_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(KeyError, match="Unknown task id"):
            scheduler.get_task("nonexistent")

    def test_tasks_property_returns_copy(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        copy = scheduler.tasks
        copy["b"] = _make_task("b")
        assert "b" not in scheduler.tasks


# =====================================================================
# 3. Basic task execution
# =====================================================================

class TestBasicExecution:
    """Tests for running tasks end-to-end."""

    @pytest.mark.asyncio
    async def test_single_task_runs_to_completion(self):
        scheduler = TaskScheduler()
        t = _make_task("a")
        scheduler.add_task(t, _noop)
        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "result-a"
        assert metrics.total_time is not None
        assert metrics.total_time >= 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        scheduler = TaskScheduler()
        for tid in ("a", "b", "c"):
            scheduler.add_task(_make_task(tid), _noop)

        await scheduler.run()
        for tid in ("a", "b", "c"):
            assert scheduler.get_task(tid).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_result_is_stored(self):
        async def custom(task: Task):
            return {"value": 42}

        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), custom)
        await scheduler.run()
        assert scheduler.get_task("a").result == {"value": 42}

    @pytest.mark.asyncio
    async def test_run_with_no_tasks(self):
        scheduler = TaskScheduler()
        metrics = await scheduler.run()
        assert metrics.total_time is not None
        assert len(metrics.task_metrics) == 0


# =====================================================================
# 4. Dependency resolution
# =====================================================================

class TestDependencyResolution:
    """Tests for topological sort and execution order."""

    def test_execution_plan_linear_chain(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        scheduler.add_task(_make_task("c", deps=["b"]), _noop)

        plan = scheduler.get_execution_plan()
        # a first, then b, then c – each in its own group
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_execution_plan_diamond(self):
        """Diamond: a -> b, a -> c, b -> d, c -> d."""
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        scheduler.add_task(_make_task("c", deps=["a"]), _noop)
        scheduler.add_task(_make_task("d", deps=["b", "c"]), _noop)

        plan = scheduler.get_execution_plan()
        assert plan[0] == ["a"]
        # b and c should be in the same group (order by priority, both 5)
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    def test_execution_plan_independent_tasks(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", priority=3), _noop)
        scheduler.add_task(_make_task("b", priority=7), _noop)
        scheduler.add_task(_make_task("c", priority=1), _noop)

        plan = scheduler.get_execution_plan()
        # All independent, should be in one group sorted by priority descending
        assert len(plan) == 1
        assert plan[0] == ["b", "a", "c"]

    @pytest.mark.asyncio
    async def test_dependency_order_respected_at_runtime(self):
        execution_order: list[str] = []

        async def track(task: Task):
            execution_order.append(task.id)
            return task.id

        scheduler = TaskScheduler(max_concurrency=1)
        scheduler.add_task(_make_task("a"), track)
        scheduler.add_task(_make_task("b", deps=["a"]), track)
        scheduler.add_task(_make_task("c", deps=["b"]), track)
        await scheduler.run()

        assert execution_order == ["a", "b", "c"]

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", deps=["nonexistent"]), _noop)
        with pytest.raises(KeyError, match="unknown task"):
            scheduler.get_execution_plan()


# =====================================================================
# 5. Circular dependency detection
# =====================================================================

class TestCircularDependency:
    """Tests for cycle detection in the dependency graph."""

    def test_simple_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", deps=["b"]), _noop)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            scheduler.get_execution_plan()

    def test_self_dependency(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", deps=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", deps=["c"]), _noop)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        scheduler.add_task(_make_task("c", deps=["b"]), _noop)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_circular_dependency_prevents_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a", deps=["b"]), _noop)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_cycle_error_message_contains_involved_tasks(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("x", deps=["y"]), _noop)
        scheduler.add_task(_make_task("y", deps=["x"]), _noop)
        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()
        msg = str(exc_info.value)
        assert "x" in msg
        assert "y" in msg


# =====================================================================
# 6. Retry logic with exponential backoff
# =====================================================================

class TestRetryLogic:
    """Tests for retry behavior and exponential backoff."""

    @pytest.mark.asyncio
    async def test_task_retries_on_failure_then_succeeds(self):
        call_count = 0

        async def flaky(task: Task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "ok"

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=3), flaky)
        await scheduler.run()

        t = scheduler.get_task("a")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "ok"
        assert t.retry_count == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self):
        async def always_fail(task: Task):
            raise RuntimeError("permanent error")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=2), always_fail)
        await scheduler.run()

        t = scheduler.get_task("a")
        assert t.status == TaskStatus.FAILED
        assert isinstance(t.result, RuntimeError)
        assert t.retry_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify that backoff delays grow exponentially."""
        timestamps: list[float] = []

        async def record_and_fail(task: Task):
            timestamps.append(time.monotonic())
            raise RuntimeError("fail")

        base = 0.05
        scheduler = TaskScheduler(backoff_base=base)
        scheduler.add_task(_make_task("a", max_retries=3), record_and_fail)
        await scheduler.run()

        # We expect 3 attempts: initial + 2 retries
        assert len(timestamps) == 3
        # Delay between attempt 1 and 2: base * 2^0 = base
        delay1 = timestamps[1] - timestamps[0]
        # Delay between attempt 2 and 3: base * 2^1 = 2*base
        delay2 = timestamps[2] - timestamps[1]

        # Allow some tolerance for async scheduling overhead
        assert delay1 >= base * 0.8
        assert delay2 >= (2 * base) * 0.8
        # Verify the second delay is roughly double the first
        assert delay2 > delay1 * 1.5

    @pytest.mark.asyncio
    async def test_zero_retries_means_fail_immediately(self):
        call_count = 0

        async def fail_once(task: Task):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=1), fail_once)
        await scheduler.run()

        assert scheduler.get_task("a").status == TaskStatus.FAILED
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self):
        call_count = 0

        async def flaky(task: Task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "ok"

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=5), flaky)
        metrics = await scheduler.run()

        assert metrics.task_metrics["a"].retry_count == 2
        assert metrics.total_retries == 2


# =====================================================================
# 7. Concurrent execution respecting concurrency limits
# =====================================================================

class TestConcurrency:
    """Tests for max_concurrency enforcement."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Ensure no more than max_concurrency tasks run simultaneously."""
        max_concurrent = 2
        peak_concurrency = 0
        current_concurrency = 0
        lock = asyncio.Lock()

        async def tracked(task: Task):
            nonlocal current_concurrency, peak_concurrency
            async with lock:
                current_concurrency += 1
                peak_concurrency = max(peak_concurrency, current_concurrency)
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrency -= 1
            return task.id

        scheduler = TaskScheduler(max_concurrency=max_concurrent)
        for i in range(6):
            scheduler.add_task(_make_task(str(i)), tracked)

        await scheduler.run()
        assert peak_concurrency <= max_concurrent

    @pytest.mark.asyncio
    async def test_concurrency_one_serializes_execution(self):
        """With max_concurrency=1, tasks should run one at a time."""
        execution_log: list[tuple[str, str]] = []

        async def track(task: Task):
            execution_log.append((task.id, "start"))
            await asyncio.sleep(0.01)
            execution_log.append((task.id, "end"))
            return task.id

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(3):
            scheduler.add_task(_make_task(str(i)), track)
        await scheduler.run()

        # With concurrency=1, every "end" should come before the next "start"
        for idx in range(len(execution_log) - 1):
            if execution_log[idx][1] == "start":
                assert execution_log[idx + 1][1] == "end"
                assert execution_log[idx + 1][0] == execution_log[idx][0]

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_high_concurrency(self):
        scheduler = TaskScheduler(max_concurrency=100)
        for i in range(20):
            scheduler.add_task(_make_task(str(i)), _sleep_short)
        await scheduler.run()
        for i in range(20):
            assert scheduler.get_task(str(i)).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrent_groups_run_in_parallel(self):
        """Independent tasks within a group should actually overlap."""
        start_times: dict[str, float] = {}
        end_times: dict[str, float] = {}

        async def tracked(task: Task):
            start_times[task.id] = time.monotonic()
            await asyncio.sleep(0.05)
            end_times[task.id] = time.monotonic()

        scheduler = TaskScheduler(max_concurrency=4)
        for tid in ("a", "b", "c", "d"):
            scheduler.add_task(_make_task(tid), tracked)
        await scheduler.run()

        # All 4 independent tasks should start before any finishes (overlap)
        latest_start = max(start_times.values())
        earliest_end = min(end_times.values())
        assert latest_start < earliest_end


# =====================================================================
# 8. Event / observer system
# =====================================================================

class TestEventSystem:
    """Tests for on / off / emit observer pattern."""

    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started: list[str] = []

        def on_start(task: Task):
            started.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on("on_task_start", on_start)
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()

        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed: list[str] = []

        def on_complete(task: Task):
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on("on_task_complete", on_complete)
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()

        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed: list[tuple[str, str]] = []

        def on_fail(task: Task, exc: Exception):
            failed.append((task.id, str(exc)))

        async def raise_err(task: Task):
            raise ValueError("boom")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.on("on_task_fail", on_fail)
        scheduler.add_task(_make_task("a", max_retries=1), raise_err)
        await scheduler.run()

        assert len(failed) == 1
        assert failed[0][0] == "a"
        assert "boom" in failed[0][1]

    @pytest.mark.asyncio
    async def test_off_removes_listener(self):
        calls: list[str] = []

        def listener(task: Task):
            calls.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on("on_task_start", listener)
        scheduler.off("on_task_start", listener)
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()

        assert calls == []

    @pytest.mark.asyncio
    async def test_off_nonexistent_listener_is_noop(self):
        scheduler = TaskScheduler()
        # Should not raise
        scheduler.off("on_task_start", lambda t: None)

    @pytest.mark.asyncio
    async def test_async_listener(self):
        completed: list[str] = []

        async def async_on_complete(task: Task):
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on("on_task_complete", async_on_complete)
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()

        assert "a" in completed

    @pytest.mark.asyncio
    async def test_multiple_listeners(self):
        log1: list[str] = []
        log2: list[str] = []

        def l1(task: Task):
            log1.append(task.id)

        def l2(task: Task):
            log2.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on("on_task_complete", l1)
        scheduler.on("on_task_complete", l2)
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()

        assert "a" in log1
        assert "a" in log2


# =====================================================================
# 9. Metrics collection
# =====================================================================

class TestMetricsCollection:
    """Tests for per-task and aggregate metrics."""

    @pytest.mark.asyncio
    async def test_metrics_populated_on_success(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _sleep_short)
        metrics = await scheduler.run()

        m = metrics.task_metrics["a"]
        assert m.task_id == "a"
        assert m.status == TaskStatus.COMPLETED
        assert m.start_time is not None
        assert m.end_time is not None
        assert m.duration is not None
        assert m.duration > 0
        assert m.retry_count == 0

    @pytest.mark.asyncio
    async def test_metrics_populated_on_failure(self):
        async def fail(task: Task):
            raise RuntimeError("fail")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=2), fail)
        metrics = await scheduler.run()

        m = metrics.task_metrics["a"]
        assert m.status == TaskStatus.FAILED
        assert m.retry_count == 2

    @pytest.mark.asyncio
    async def test_total_time_is_reasonable(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        metrics = await scheduler.run()
        assert metrics.total_time is not None
        assert metrics.total_time >= 0
        # Running a noop should not take more than a few seconds
        assert metrics.total_time < 5.0

    @pytest.mark.asyncio
    async def test_metrics_property(self):
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)
        await scheduler.run()
        # The metrics property should return the same object
        assert scheduler.metrics.total_time is not None


# =====================================================================
# 10. Edge cases: failed dependency propagation, complex graphs
# =====================================================================

class TestEdgeCases:
    """Tests for failure propagation, mixed scenarios, and boundary conditions."""

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades(self):
        """If a dependency fails, its dependents should be marked FAILED."""
        async def fail_a(task: Task):
            raise RuntimeError("a failed")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=1), fail_a)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        scheduler.add_task(_make_task("c", deps=["b"]), _noop)
        await scheduler.run()

        assert scheduler.get_task("a").status == TaskStatus.FAILED
        assert scheduler.get_task("b").status == TaskStatus.FAILED
        assert scheduler.get_task("c").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_partial_failure_doesnt_block_independent_tasks(self):
        """Tasks without failed deps should still run even if other tasks fail."""
        async def fail(task: Task):
            raise RuntimeError("fail")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.add_task(_make_task("a", max_retries=1), fail)
        scheduler.add_task(_make_task("b"), _noop)  # independent
        await scheduler.run()

        assert scheduler.get_task("a").status == TaskStatus.FAILED
        assert scheduler.get_task("b").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_failed_dependency_emits_fail_event(self):
        failed_ids: list[str] = []

        def on_fail(task: Task, exc: Exception):
            failed_ids.append(task.id)

        async def fail(task: Task):
            raise RuntimeError("fail")

        scheduler = TaskScheduler(backoff_base=0.001)
        scheduler.on("on_task_fail", on_fail)
        scheduler.add_task(_make_task("a", max_retries=1), fail)
        scheduler.add_task(_make_task("b", deps=["a"]), _noop)
        await scheduler.run()

        assert "a" in failed_ids
        assert "b" in failed_ids

    @pytest.mark.asyncio
    async def test_wide_dependency_tree(self):
        """Many tasks depending on a single root."""
        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(_make_task("root"), _noop)
        for i in range(10):
            scheduler.add_task(_make_task(f"child-{i}", deps=["root"]), _noop)
        await scheduler.run()

        assert scheduler.get_task("root").status == TaskStatus.COMPLETED
        for i in range(10):
            assert scheduler.get_task(f"child-{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_deep_dependency_chain(self):
        """A -> B -> C -> D -> ... (10 levels deep)."""
        scheduler = TaskScheduler(max_concurrency=2)
        prev = None
        for i in range(10):
            tid = f"t{i}"
            deps = [prev] if prev else []
            scheduler.add_task(_make_task(tid, deps=deps), _noop)
            prev = tid
        await scheduler.run()

        for i in range(10):
            assert scheduler.get_task(f"t{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_priority_ordering_within_group(self):
        """Higher-priority tasks should appear first in execution plan groups."""
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("low", priority=1), _noop)
        scheduler.add_task(_make_task("mid", priority=5), _noop)
        scheduler.add_task(_make_task("high", priority=10), _noop)

        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert plan[0] == ["high", "mid", "low"]

    @pytest.mark.asyncio
    async def test_scheduler_resets_metrics_on_rerun(self):
        """Running the scheduler again should produce fresh metrics."""
        scheduler = TaskScheduler()
        scheduler.add_task(_make_task("a"), _noop)

        m1 = await scheduler.run()
        assert "a" in m1.task_metrics

        # Reset task status for a re-run (tasks remain registered)
        scheduler.get_task("a").status = TaskStatus.PENDING
        scheduler.get_task("a").retry_count = 0

        m2 = await scheduler.run()
        assert m2 is not m1
        assert "a" in m2.task_metrics


# =====================================================================
# 11. TaskStatus enum
# =====================================================================

class TestTaskStatus:
    """Tests for the TaskStatus enum values."""

    def test_all_statuses_exist(self):
        assert TaskStatus.PENDING.value == "PENDING"
        assert TaskStatus.RUNNING.value == "RUNNING"
        assert TaskStatus.COMPLETED.value == "COMPLETED"
        assert TaskStatus.FAILED.value == "FAILED"
