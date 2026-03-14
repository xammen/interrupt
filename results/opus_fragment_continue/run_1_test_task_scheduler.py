"""
Comprehensive tests for task_scheduler.py.

Covers: Task validation, dependency resolution, circular-dependency detection,
concurrency control, retry logic with exponential backoff, observer events,
execution plan, scheduler metrics, edge cases, and reset behaviour.
"""

from __future__ import annotations

import asyncio
import time

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
    """Trivial coroutine that just returns a string."""
    return f"{task.id}-done"


async def _slow(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate real work."""
    await asyncio.sleep(0.05)
    return f"{task.id}-slow-done"


def _make_failing_coro(fail_times: int = 1):
    """Return a coroutine factory that fails *fail_times* then succeeds."""
    call_count = 0

    async def _coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count}")
        return f"{task.id}-recovered"

    return _coro


# ===========================================================================
# Task dataclass tests
# ===========================================================================

class TestTask:
    def test_default_values(self):
        t = Task(id="t1", name="Task 1")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_boundaries(self):
        Task(id="low", name="Low", priority=1)
        Task(id="high", name="High", priority=10)

    def test_priority_too_low(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="Bad", priority=0)

    def test_priority_too_high(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="Bad", priority=11)

    def test_created_at_populated(self):
        t = Task(id="t1", name="Task 1")
        assert t.created_at is not None


# ===========================================================================
# TaskMetrics tests
# ===========================================================================

class TestTaskMetrics:
    def test_elapsed(self):
        m = TaskMetrics(task_id="x", start_time=10.0, end_time=12.5)
        assert m.elapsed == pytest.approx(2.5)

    def test_elapsed_zero(self):
        m = TaskMetrics(task_id="x")
        assert m.elapsed == 0.0


# ===========================================================================
# Scheduler construction
# ===========================================================================

class TestSchedulerInit:
    def test_default_concurrency(self):
        s = TaskScheduler()
        assert s._max_concurrency == 4

    def test_custom_concurrency(self):
        s = TaskScheduler(max_concurrency=8)
        assert s._max_concurrency == 8

    def test_invalid_concurrency(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)


# ===========================================================================
# Task registration
# ===========================================================================

class TestAddTask:
    def test_add_single_task(self):
        s = TaskScheduler()
        t = Task(id="a", name="A")
        s.add_task(t, _noop)
        assert "a" in s.tasks

    def test_duplicate_task_id(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="A2"), _noop)

    def test_tasks_property_returns_copy(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        copy = s.tasks
        copy["b"] = Task(id="b", name="B")
        assert "b" not in s.tasks

    def test_get_task(self):
        s = TaskScheduler()
        t = Task(id="a", name="A")
        s.add_task(t, _noop)
        assert s.get_task("a") is t

    def test_get_task_missing(self):
        s = TaskScheduler()
        with pytest.raises(KeyError):
            s.get_task("nonexistent")


# ===========================================================================
# Dependency validation
# ===========================================================================

class TestDependencyValidation:
    def test_unknown_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        with pytest.raises(ValueError, match="unknown task 'b'"):
            s.get_execution_plan()

    def test_valid_dependency(self):
        s = TaskScheduler()
        s.add_task(Task(id="b", name="B"), _noop)
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 2


# ===========================================================================
# Circular dependency detection
# ===========================================================================

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
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["c"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert exc_info.value.cycle is not None

    def test_error_message_contains_cycle_path(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X", dependencies=["y"]), _noop)
        s.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop)
        with pytest.raises(CircularDependencyError, match="->"):
            s.get_execution_plan()

    def test_no_cycle_message(self):
        err = CircularDependencyError()
        assert "Circular dependency detected in the task graph" in str(err)
        assert err.cycle is None


# ===========================================================================
# Execution plan (topological sort + priority)
# ===========================================================================

class TestExecutionPlan:
    def test_single_task(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        assert s.get_execution_plan() == [["a"]]

    def test_linear_chain(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_independent_tasks_same_tier(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B"), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_priority_ordering_within_tier(self):
        s = TaskScheduler()
        s.add_task(Task(id="low", name="Low", priority=1), _noop)
        s.add_task(Task(id="high", name="High", priority=10), _noop)
        s.add_task(Task(id="mid", name="Mid", priority=5), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert plan[0] == ["high", "mid", "low"]

    def test_diamond_dependency(self):
        """
            a
           / \\
          b   c
           \\ /
            d
        """
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _noop)
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), _noop)
        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    def test_empty_scheduler(self):
        s = TaskScheduler()
        assert s.get_execution_plan() == []


# ===========================================================================
# Scheduler execution (run)
# ===========================================================================

class TestSchedulerRun:
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
    async def test_dependency_chain_execution_order(self):
        """Tasks run in dependency order: a -> b -> c."""
        order: list[str] = []

        async def _track(task: Task) -> str:
            order.append(task.id)
            return task.id

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _track)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _track)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _track)
        await s.run()

        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Two independent tasks should overlap in wall-clock time."""
        s = TaskScheduler(max_concurrency=2)
        s.add_task(Task(id="a", name="A"), _slow)
        s.add_task(Task(id="b", name="B"), _slow)

        start = time.monotonic()
        await s.run()
        elapsed = time.monotonic() - start

        # Both should finish roughly together (~0.05s), not serially (~0.1s)
        assert elapsed < 0.09

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """With concurrency=1, tasks should run serially."""
        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="a", name="A"), _slow)
        s.add_task(Task(id="b", name="B"), _slow)

        start = time.monotonic()
        await s.run()
        elapsed = time.monotonic() - start

        # Serial execution: at least ~0.1s
        assert elapsed >= 0.09

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades(self):
        """If task 'a' fails, dependent task 'b' should also fail."""
        async def _always_fail(task: Task) -> None:
            raise RuntimeError("boom")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), _always_fail)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        await s.run()

        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_multiple_tasks_results(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B"), _noop)
        await s.run()

        assert s.get_task("a").result == "a-done"
        assert s.get_task("b").result == "b-done"


# ===========================================================================
# Retry logic
# ===========================================================================

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        coro = _make_failing_coro(fail_times=2)
        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=3), coro)
        await s.run()

        t = s.get_task("r")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "r-recovered"
        assert t.retry_count == 2

    @pytest.mark.asyncio
    async def test_exceed_max_retries(self):
        coro = _make_failing_coro(fail_times=5)
        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=2), coro)
        await s.run()

        t = s.get_task("r")
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # tried original + 2 retries => 3 total

    @pytest.mark.asyncio
    async def test_no_retries_on_zero_max(self):
        async def _fail(task: Task) -> None:
            raise RuntimeError("fail")

        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=0), _fail)
        await s.run()

        t = s.get_task("r")
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_metrics_recorded(self):
        coro = _make_failing_coro(fail_times=1)
        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=3), coro)
        metrics = await s.run()

        assert metrics.task_metrics["r"].retries == 1


# ===========================================================================
# Observer / event system
# ===========================================================================

class TestObserverEvents:
    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_start", lambda task: started.append(task.id))
        await s.run()
        assert started == ["a"]

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", lambda task: completed.append(task.id))
        await s.run()
        assert completed == ["a"]

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed: list[str] = []

        async def _fail(task: Task) -> None:
            raise RuntimeError("fail")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), _fail)
        s.on("on_task_fail", lambda task, exc: failed.append(task.id))
        await s.run()
        assert failed == ["a"]

    @pytest.mark.asyncio
    async def test_async_callback(self):
        started: list[str] = []

        async def _on_start(task: Task) -> None:
            started.append(task.id)

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_start", _on_start)
        await s.run()
        assert started == ["a"]

    @pytest.mark.asyncio
    async def test_multiple_listeners(self):
        log1: list[str] = []
        log2: list[str] = []

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", lambda task: log1.append(task.id))
        s.on("on_task_complete", lambda task: log2.append(task.id))
        await s.run()

        assert log1 == ["a"]
        assert log2 == ["a"]

    @pytest.mark.asyncio
    async def test_cascade_failure_emits_on_task_fail(self):
        """Dependent task that is skipped due to dep failure should emit on_task_fail."""
        fail_ids: list[str] = []

        async def _fail(task: Task) -> None:
            raise RuntimeError("boom")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), _fail)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.on("on_task_fail", lambda task, exc: fail_ids.append(task.id))
        await s.run()

        assert "a" in fail_ids
        assert "b" in fail_ids


# ===========================================================================
# Scheduler metrics
# ===========================================================================

class TestSchedulerMetrics:
    @pytest.mark.asyncio
    async def test_total_time(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        metrics = await s.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_per_task_metrics(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B"), _noop)
        metrics = await s.run()

        assert "a" in metrics.task_metrics
        assert "b" in metrics.task_metrics
        assert metrics.task_metrics["a"].elapsed >= 0
        assert metrics.task_metrics["b"].elapsed >= 0

    def test_scheduler_metrics_defaults(self):
        m = SchedulerMetrics()
        assert m.total_time == 0.0
        assert m.task_metrics == {}


# ===========================================================================
# Reset
# ===========================================================================

class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        s.reset()

        assert s.get_task("a").status == TaskStatus.PENDING
        assert s.get_task("a").result is None
        assert s.get_task("a").retry_count == 0

    @pytest.mark.asyncio
    async def test_can_rerun_after_reset(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        await s.run()
        s.reset()
        metrics = await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert metrics.total_time > 0


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_run(self):
        s = TaskScheduler()
        metrics = await s.run()
        assert metrics.total_time >= 0
        assert metrics.task_metrics == {}

    @pytest.mark.asyncio
    async def test_large_fan_out(self):
        """One root task with many dependents."""
        s = TaskScheduler(max_concurrency=10)
        s.add_task(Task(id="root", name="Root"), _noop)
        for i in range(20):
            s.add_task(
                Task(id=f"child_{i}", name=f"Child {i}", dependencies=["root"]),
                _noop,
            )
        await s.run()

        assert s.get_task("root").status == TaskStatus.COMPLETED
        for i in range(20):
            assert s.get_task(f"child_{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_returning_none(self):
        async def _none(task: Task) -> None:
            return None

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _none)
        await s.run()
        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result is None
