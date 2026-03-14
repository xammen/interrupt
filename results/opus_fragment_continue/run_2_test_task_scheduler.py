"""Tests for the async task scheduler module."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from task_scheduler import (
    CircularDependencyError,
    TaskNotFoundError,
    TaskStatus,
    Task,
    TaskMetrics,
    SchedulerMetrics,
    TaskScheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_coro(task: Task) -> str:
    """A simple coroutine that returns the task's name."""
    return f"done-{task.name}"


async def _slow_coro(task: Task) -> str:
    """A coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"slow-{task.name}"


async def _failing_coro(task: Task) -> str:
    """A coroutine that always raises."""
    raise RuntimeError(f"boom-{task.name}")


_fail_then_pass_counters: dict[str, int] = {}


async def _fail_then_pass_coro(task: Task) -> str:
    """Fails on the first call, succeeds on subsequent calls."""
    _fail_then_pass_counters.setdefault(task.id, 0)
    _fail_then_pass_counters[task.id] += 1
    if _fail_then_pass_counters[task.id] <= 1:
        raise RuntimeError("transient failure")
    return "recovered"


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------


class TestTask:
    def test_create_default(self) -> None:
        t = Task(id="a", name="Alpha")
        assert t.id == "a"
        assert t.name == "Alpha"
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_boundary_valid(self) -> None:
        Task(id="low", name="Low", priority=1)
        Task(id="high", name="High", priority=10)

    def test_priority_too_low(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="X", priority=0)

    def test_priority_too_high(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="X", priority=11)

    def test_dependencies_list(self) -> None:
        t = Task(id="b", name="Beta", dependencies=["a"])
        assert t.dependencies == ["a"]


# ---------------------------------------------------------------------------
# TaskMetrics tests
# ---------------------------------------------------------------------------


class TestTaskMetrics:
    def test_duration_none_when_incomplete(self) -> None:
        m = TaskMetrics(task_id="t")
        assert m.duration is None
        m.start_time = 1.0
        assert m.duration is None

    def test_duration_calculated(self) -> None:
        m = TaskMetrics(task_id="t", start_time=10.0, end_time=12.5)
        assert m.duration == pytest.approx(2.5)


class TestSchedulerMetrics:
    def test_total_time_none_when_incomplete(self) -> None:
        sm = SchedulerMetrics()
        assert sm.total_time is None

    def test_total_time_calculated(self) -> None:
        sm = SchedulerMetrics(total_start=0.0, total_end=5.0)
        assert sm.total_time == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TaskScheduler — task management
# ---------------------------------------------------------------------------


class TestSchedulerTaskManagement:
    def test_add_and_get_task(self) -> None:
        sched = TaskScheduler()
        t = Task(id="a", name="Alpha")
        sched.add_task(t, _noop_coro)
        assert sched.get_task("a") is t
        assert sched.task_count == 1

    def test_add_duplicate_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        with pytest.raises(ValueError, match="already exists"):
            sched.add_task(Task(id="a", name="A2"), _noop_coro)

    def test_get_missing_raises(self) -> None:
        sched = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            sched.get_task("nope")

    def test_remove_task(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        removed = sched.remove_task("a")
        assert removed.id == "a"
        assert sched.task_count == 1
        # dependency reference should be cleaned up
        assert "a" not in sched.get_task("b").dependencies

    def test_remove_missing_raises(self) -> None:
        sched = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            sched.remove_task("nope")

    def test_get_tasks_by_status(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        sched.add_task(Task(id="b", name="B"), _noop_coro)
        pending = sched.get_tasks_by_status(TaskStatus.PENDING)
        assert len(pending) == 2


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(
            Task(id="a", name="A", dependencies=["missing"]), _noop_coro
        )
        with pytest.raises(TaskNotFoundError, match="unknown task 'missing'"):
            await sched.run()

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", dependencies=["b"]), _noop_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    @pytest.mark.asyncio
    async def test_self_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", dependencies=["a"]), _noop_coro)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    @pytest.mark.asyncio
    async def test_three_node_cycle(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", dependencies=["c"]), _noop_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        sched.add_task(Task(id="c", name="C", dependencies=["b"]), _noop_coro)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    def test_topological_layers_linear_chain(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        sched.add_task(Task(id="c", name="C", dependencies=["b"]), _noop_coro)
        layers = sched._topological_sort()
        assert len(layers) == 3
        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_topological_layers_diamond(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"], priority=8), _noop_coro)
        sched.add_task(Task(id="c", name="C", dependencies=["a"], priority=3), _noop_coro)
        sched.add_task(Task(id="d", name="D", dependencies=["b", "c"]), _noop_coro)
        layers = sched._topological_sort()
        assert len(layers) == 3
        assert layers[0] == ["a"]
        # b has higher priority, should come first in the layer
        assert layers[1] == ["b", "c"]
        assert layers[2] == ["d"]


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_exponential_growth(self) -> None:
        assert TaskScheduler._backoff_delay(0, base=0.1) == pytest.approx(0.1)
        assert TaskScheduler._backoff_delay(1, base=0.1) == pytest.approx(0.2)
        assert TaskScheduler._backoff_delay(2, base=0.1) == pytest.approx(0.4)

    def test_cap(self) -> None:
        assert TaskScheduler._backoff_delay(100, base=0.1, cap=5.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Observer / events
# ---------------------------------------------------------------------------


class TestObserver:
    @pytest.mark.asyncio
    async def test_on_task_start_fires(self) -> None:
        sched = TaskScheduler()
        events: list[str] = []
        sched.on("on_task_start", lambda task: events.append(f"start:{task.id}"))
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()
        assert "start:a" in events

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self) -> None:
        sched = TaskScheduler()
        events: list[str] = []
        sched.on("on_task_complete", lambda task: events.append(f"done:{task.id}"))
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()
        assert "done:a" in events

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self) -> None:
        sched = TaskScheduler()
        events: list[tuple[str, str]] = []
        sched.on(
            "on_task_fail",
            lambda task, exc: events.append((task.id, str(exc))),
        )
        sched.add_task(Task(id="a", name="A", max_retries=0), _failing_coro)
        await sched.run()
        assert len(events) == 1
        assert events[0][0] == "a"

    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        sched = TaskScheduler()
        events: list[str] = []

        async def async_cb(task: Task) -> None:
            events.append(f"async:{task.id}")

        sched.on("on_task_complete", async_cb)
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()
        assert "async:a" in events


# ---------------------------------------------------------------------------
# Run — basic execution
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_empty_scheduler(self) -> None:
        sched = TaskScheduler()
        result = await sched.run()
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_task(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        result = await sched.run()
        assert result["a"].status == TaskStatus.COMPLETED
        assert result["a"].result == "done-A"

    @pytest.mark.asyncio
    async def test_linear_chain(self) -> None:
        sched = TaskScheduler()
        execution_order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            execution_order.append(task.id)
            return task.id

        sched.add_task(Task(id="a", name="A"), tracking_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), tracking_coro)
        sched.add_task(Task(id="c", name="C", dependencies=["b"]), tracking_coro)
        result = await sched.run()

        # All completed
        for tid in ["a", "b", "c"]:
            assert result[tid].status == TaskStatus.COMPLETED

        # Order must respect dependencies
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("b") < execution_order.index("c")

    @pytest.mark.asyncio
    async def test_parallel_independent_tasks(self) -> None:
        sched = TaskScheduler(max_concurrency=4)
        timestamps: dict[str, float] = {}

        async def stamp_coro(task: Task) -> str:
            import time as _t
            timestamps[task.id] = _t.monotonic()
            await asyncio.sleep(0.1)
            return task.id

        for i in range(4):
            sched.add_task(Task(id=f"t{i}", name=f"T{i}"), stamp_coro)

        await sched.run()

        # All tasks should have started nearly simultaneously
        starts = list(timestamps.values())
        assert max(starts) - min(starts) < 0.08  # generous tolerance

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_coro(task: Task) -> str:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return task.id

        sched = TaskScheduler(max_concurrency=2)
        for i in range(6):
            sched.add_task(Task(id=f"t{i}", name=f"T{i}"), counting_coro)
        await sched.run()
        assert max_concurrent <= 2


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self) -> None:
        _fail_then_pass_counters.clear()
        sched = TaskScheduler()
        sched.add_task(
            Task(id="a", name="A", max_retries=3), _fail_then_pass_coro
        )
        result = await sched.run()
        assert result["a"].status == TaskStatus.COMPLETED
        assert result["a"].result == "recovered"
        assert result["a"].retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", max_retries=2), _failing_coro)
        result = await sched.run()
        assert result["a"].status == TaskStatus.FAILED
        assert result["a"].retry_count == 3  # initial + 2 retries = 3 attempts

    @pytest.mark.asyncio
    async def test_no_retries(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", max_retries=0), _failing_coro)
        result = await sched.run()
        assert result["a"].status == TaskStatus.FAILED
        assert result["a"].retry_count == 1


# ---------------------------------------------------------------------------
# Failed dependency propagation
# ---------------------------------------------------------------------------


class TestFailedDependencyPropagation:
    @pytest.mark.asyncio
    async def test_dependent_fails_when_dependency_fails(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", max_retries=0), _failing_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        result = await sched.run()
        assert result["a"].status == TaskStatus.FAILED
        assert result["b"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_unrelated_task_still_runs(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A", max_retries=0), _failing_coro)
        sched.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro)
        sched.add_task(Task(id="c", name="C"), _noop_coro)
        result = await sched.run()
        assert result["a"].status == TaskStatus.FAILED
        assert result["b"].status == TaskStatus.FAILED
        assert result["c"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_populated(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()

        assert sched.metrics.total_time is not None
        assert sched.metrics.total_time >= 0
        assert "a" in sched.metrics.task_metrics
        tm = sched.metrics.task_metrics["a"]
        assert tm.duration is not None
        assert tm.duration >= 0

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self) -> None:
        _fail_then_pass_counters.clear()
        sched = TaskScheduler()
        sched.add_task(
            Task(id="a", name="A", max_retries=3), _fail_then_pass_coro
        )
        await sched.run()
        assert sched.metrics.task_metrics["a"].retries == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()
        assert sched.get_task("a").status == TaskStatus.COMPLETED

        sched.reset()
        assert sched.get_task("a").status == TaskStatus.PENDING
        assert sched.get_task("a").result is None
        assert sched.get_task("a").retry_count == 0
        assert sched.metrics.total_time is None

    @pytest.mark.asyncio
    async def test_can_rerun_after_reset(self) -> None:
        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="A"), _noop_coro)
        await sched.run()
        sched.reset()
        result = await sched.run()
        assert result["a"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriority:
    @pytest.mark.asyncio
    async def test_higher_priority_runs_first_in_layer(self) -> None:
        sched = TaskScheduler(max_concurrency=1)
        execution_order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            execution_order.append(task.id)
            return task.id

        # All independent — same layer, but concurrency=1 forces serial
        sched.add_task(Task(id="low", name="Low", priority=1), tracking_coro)
        sched.add_task(Task(id="mid", name="Mid", priority=5), tracking_coro)
        sched.add_task(Task(id="high", name="High", priority=10), tracking_coro)
        await sched.run()

        # Higher priority should come first
        assert execution_order.index("high") < execution_order.index("mid")
        assert execution_order.index("mid") < execution_order.index("low")


# ---------------------------------------------------------------------------
# CircularDependencyError
# ---------------------------------------------------------------------------


class TestCircularDependencyError:
    def test_message_format(self) -> None:
        err = CircularDependencyError(["a", "b", "c", "a"])
        assert "a -> b -> c -> a" in str(err)
        assert err.cycle == ["a", "b", "c", "a"]
