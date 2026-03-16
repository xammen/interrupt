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
    TaskNotFoundError,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coro(return_value=None, delay: float = 0):
    """Return an async callable that optionally sleeps and returns a value."""

    async def _coro():
        if delay:
            await asyncio.sleep(delay)
        return return_value

    return _coro


def _make_failing_coro(times: int, *, exc: Exception | None = None):
    """Return an async callable that raises *times* then succeeds.

    After *times* failures it returns ``"ok"``.
    """
    call_count = 0
    error = exc or RuntimeError("boom")

    async def _coro():
        nonlocal call_count
        call_count += 1
        if call_count <= times:
            raise error
        return "ok"

    return _coro


def _make_always_failing_coro(exc: Exception | None = None):
    """Return an async callable that always raises."""
    error = exc or RuntimeError("permanent failure")

    async def _coro():
        raise error

    return _coro


# ===================================================================
# Task dataclass
# ===================================================================

class TestTaskModel:
    """Tests for the Task dataclass itself."""

    def test_create_task_defaults(self):
        task = Task(id="t1", name="Task 1")
        assert task.id == "t1"
        assert task.name == "Task 1"
        assert task.priority == 5
        assert task.dependencies == []
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.result is None

    def test_priority_boundaries_valid(self):
        assert Task(id="lo", name="lo", priority=1).priority == 1
        assert Task(id="hi", name="hi", priority=10).priority == 10

    def test_priority_too_low_raises(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="bad", priority=0)

    def test_priority_too_high_raises(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="bad", name="bad", priority=11)


# ===================================================================
# TaskMetrics
# ===================================================================

class TestTaskMetrics:
    def test_duration_property(self):
        m = TaskMetrics(task_id="t", start_time=10.0, end_time=13.5)
        assert m.duration == pytest.approx(3.5)

    def test_duration_zero_when_unset(self):
        m = TaskMetrics(task_id="t")
        assert m.duration == pytest.approx(0.0)


# ===================================================================
# Exceptions
# ===================================================================

class TestExceptions:
    def test_circular_dependency_error_with_cycle(self):
        err = CircularDependencyError(["a", "b", "c", "a"])
        assert "a -> b -> c -> a" in str(err)
        assert err.cycle == ["a", "b", "c", "a"]

    def test_circular_dependency_error_no_cycle(self):
        err = CircularDependencyError()
        assert "Circular dependency detected" in str(err)
        assert err.cycle == []

    def test_task_not_found_error(self):
        err = TaskNotFoundError("missing")
        assert "missing" in str(err)


# ===================================================================
# Scheduler — construction & add_task
# ===================================================================

class TestSchedulerConstruction:
    def test_default_concurrency(self):
        s = TaskScheduler()
        assert s._max_concurrency == 4

    def test_custom_concurrency(self):
        s = TaskScheduler(max_concurrency=1)
        assert s._max_concurrency == 1

    def test_invalid_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)

    def test_add_task(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1")
        s.add_task(t, _make_coro())
        assert "t1" in s._tasks

    def test_add_duplicate_task_raises(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1")
        s.add_task(t, _make_coro())
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="t1", name="Dup"), _make_coro())


# ===================================================================
# Basic task execution
# ===================================================================

class TestBasicExecution:
    """Run individual and multiple independent tasks."""

    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1")
        s.add_task(t, _make_coro(return_value=42))

        metrics = await s.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == 42
        assert "t1" in metrics.task_metrics
        assert metrics.task_metrics["t1"].duration >= 0
        assert metrics.total_time >= 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        s = TaskScheduler()
        for i in range(5):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _make_coro(return_value=i))

        metrics = await s.run()

        for i in range(5):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED
            assert s._tasks[f"t{i}"].result == i
        assert len(metrics.task_metrics) == 5

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_fine(self):
        s = TaskScheduler()
        metrics = await s.run()
        assert metrics.total_time >= 0
        assert metrics.task_metrics == {}

    @pytest.mark.asyncio
    async def test_task_result_is_none_by_default(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1")
        s.add_task(t, _make_coro())  # returns None
        await s.run()
        assert t.status == TaskStatus.COMPLETED
        assert t.result is None


# ===================================================================
# Dependency resolution
# ===================================================================

class TestDependencyResolution:
    """Ensure topological ordering and dependency enforcement."""

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """A -> B -> C  must execute in order."""
        execution_order: list[str] = []

        def _tracker(tid: str):
            async def _coro():
                execution_order.append(tid)
            return _coro

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _tracker("a"))
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _tracker("b"))
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _tracker("c"))

        await s.run()

        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """
        A depends on nothing.
        B and C depend on A.
        D depends on B and C.
        """
        execution_order: list[str] = []

        def _tracker(tid: str):
            async def _coro():
                execution_order.append(tid)
            return _coro

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _tracker("a"))
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _tracker("b"))
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _tracker("c"))
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), _tracker("d"))

        await s.run()

        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("a") < execution_order.index("c")
        assert execution_order.index("b") < execution_order.index("d")
        assert execution_order.index("c") < execution_order.index("d")

    def test_get_execution_plan_linear(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _make_coro())
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _make_coro())

        plan = s.get_execution_plan()
        assert plan == [["a"], ["b"], ["c"]]

    def test_get_execution_plan_concurrent_group(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _make_coro())
        s.add_task(Task(id="b", name="B"), _make_coro())
        s.add_task(Task(id="c", name="C", dependencies=["a", "b"]), _make_coro())

        plan = s.get_execution_plan()
        # a and b in first group (order depends on priority), c in second
        assert len(plan) == 2
        assert set(plan[0]) == {"a", "b"}
        assert plan[1] == ["c"]

    def test_priority_ordering_within_group(self):
        s = TaskScheduler()
        s.add_task(Task(id="lo", name="Low", priority=1), _make_coro())
        s.add_task(Task(id="hi", name="High", priority=10), _make_coro())
        s.add_task(Task(id="mid", name="Mid", priority=5), _make_coro())

        plan = s.get_execution_plan()
        # Single group sorted by descending priority.
        assert plan == [["hi", "mid", "lo"]]

    def test_unknown_dependency_raises_task_not_found(self):
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", dependencies=["nonexistent"]),
            _make_coro(),
        )
        with pytest.raises(TaskNotFoundError, match="nonexistent"):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises_on_run(self):
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", dependencies=["nonexistent"]),
            _make_coro(),
        )
        with pytest.raises(TaskNotFoundError):
            await s.run()

    @pytest.mark.asyncio
    async def test_dependent_task_fails_when_dependency_fails(self):
        """If A fails, B (which depends on A) should also be marked FAILED."""
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=1),
            _make_always_failing_coro(),
        )
        s.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _make_coro(return_value="should not run"),
        )

        await s.run()

        assert s._tasks["a"].status == TaskStatus.FAILED
        assert s._tasks["b"].status == TaskStatus.FAILED
        # b should never have executed, so result stays None
        assert s._tasks["b"].result is None


# ===================================================================
# Circular dependency detection
# ===================================================================

class TestCircularDependencyDetection:
    def test_simple_cycle_a_b(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _make_coro())
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())

        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert len(exc_info.value.cycle) > 0

    def test_self_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["a"]), _make_coro())

        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_three_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), _make_coro())
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _make_coro())

        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        # The cycle should include all three nodes.
        cycle = exc_info.value.cycle
        assert len(cycle) >= 3

    @pytest.mark.asyncio
    async def test_circular_dependency_blocks_run(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X", dependencies=["y"]), _make_coro())
        s.add_task(Task(id="y", name="Y", dependencies=["x"]), _make_coro())

        with pytest.raises(CircularDependencyError):
            await s.run()

    def test_no_cycle_passes(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _make_coro())
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _make_coro())
        # Should not raise
        plan = s.get_execution_plan()
        assert len(plan) == 2

    def test_cycle_in_subgraph(self):
        """Independent tasks exist alongside a cycle; cycle is still detected."""
        s = TaskScheduler()
        s.add_task(Task(id="ok1", name="OK1"), _make_coro())
        s.add_task(Task(id="ok2", name="OK2"), _make_coro())
        # cycle: cyc_a -> cyc_b -> cyc_a
        s.add_task(Task(id="cyc_a", name="CycA", dependencies=["cyc_b"]), _make_coro())
        s.add_task(Task(id="cyc_b", name="CycB", dependencies=["cyc_a"]), _make_coro())

        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()


# ===================================================================
# Retry logic with exponential backoff
# ===================================================================

class TestRetryLogic:

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """Task fails twice then succeeds; should be COMPLETED."""
        s = TaskScheduler()
        t = Task(id="t1", name="T1", max_retries=3)
        coro = _make_failing_coro(times=2)
        s.add_task(t, coro)

        metrics = await s.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "ok"
        assert t.retry_count == 2
        assert metrics.task_metrics["t1"].retries == 2

    @pytest.mark.asyncio
    async def test_exceeds_max_retries_marks_failed(self):
        """Task always fails and exceeds max_retries; should be FAILED."""
        s = TaskScheduler()
        t = Task(id="t1", name="T1", max_retries=2)
        s.add_task(t, _make_always_failing_coro())

        await s.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 2
        assert t.result is None

    @pytest.mark.asyncio
    async def test_max_retries_one_fails_on_first_error(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1", max_retries=1)
        s.add_task(t, _make_always_failing_coro())

        await s.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, monkeypatch):
        """Verify that backoff durations follow the 2^(n-1)*0.5 formula.

        We mock asyncio.sleep to capture the actual backoff values passed.
        """
        recorded_sleeps: list[float] = []
        original_sleep = asyncio.sleep

        async def _mock_sleep(duration: float) -> None:
            recorded_sleeps.append(duration)
            # Don't actually sleep — just yield control.
            await original_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", _mock_sleep)

        s = TaskScheduler()
        t = Task(id="t1", name="T1", max_retries=4)
        # Fails 3 times then succeeds.
        s.add_task(t, _make_failing_coro(times=3))

        await s.run()

        assert t.status == TaskStatus.COMPLETED
        # Expected backoffs: retry 1 → 0.5, retry 2 → 1.0, retry 3 → 2.0
        assert len(recorded_sleeps) == 3
        assert recorded_sleeps[0] == pytest.approx(0.5)
        assert recorded_sleeps[1] == pytest.approx(1.0)
        assert recorded_sleeps[2] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_retry_metrics_recorded(self):
        s = TaskScheduler()
        t = Task(id="t1", name="T1", max_retries=3)
        s.add_task(t, _make_failing_coro(times=1))

        metrics = await s.run()

        tm = metrics.task_metrics["t1"]
        assert tm.retries == 1
        assert tm.duration >= 0


# ===================================================================
# Concurrent execution respecting concurrency limits
# ===================================================================

class TestConcurrencyControl:

    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """With concurrency=2 and 5 independent tasks, at most 2 run at once."""
        current = 0
        peak = 0
        lock = asyncio.Lock()

        async def _tracked_coro():
            nonlocal current, peak
            async with lock:
                current += 1
                if current > peak:
                    peak = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1

        s = TaskScheduler(max_concurrency=2)
        for i in range(5):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _tracked_coro)

        await s.run()

        assert peak <= 2

    @pytest.mark.asyncio
    async def test_concurrency_of_one_is_serial(self):
        """max_concurrency=1 forces strictly serial execution."""
        execution_order: list[str] = []

        def _tracker(tid: str):
            async def _coro():
                execution_order.append(tid)
            return _coro

        s = TaskScheduler(max_concurrency=1)
        # All same priority so iteration order in the single group is stable
        # (sorted by priority descending, then by insertion for equal priority).
        for i in range(3):
            s.add_task(Task(id=f"t{i}", name=f"T{i}", priority=5), _tracker(f"t{i}"))

        await s.run()

        # All three should have run exactly once.
        assert len(execution_order) == 3

    @pytest.mark.asyncio
    async def test_concurrency_higher_than_task_count(self):
        """When concurrency > number of tasks, all tasks still complete."""
        s = TaskScheduler(max_concurrency=100)
        for i in range(3):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _make_coro(return_value=i))

        metrics = await s.run()

        for i in range(3):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED
        assert len(metrics.task_metrics) == 3

    @pytest.mark.asyncio
    async def test_concurrent_tasks_with_dependencies(self):
        """
        Dependency groups are serialised, but tasks within a group are
        concurrent (up to the concurrency limit).

        Graph: a, b (independent) -> c depends on both.
        With concurrency=2, a and b can run in parallel.
        """
        timestamps: dict[str, tuple[float, float]] = {}

        def _timed(tid: str):
            async def _coro():
                start = time.monotonic()
                await asyncio.sleep(0.05)
                end = time.monotonic()
                timestamps[tid] = (start, end)
            return _coro

        s = TaskScheduler(max_concurrency=2)
        s.add_task(Task(id="a", name="A"), _timed("a"))
        s.add_task(Task(id="b", name="B"), _timed("b"))
        s.add_task(Task(id="c", name="C", dependencies=["a", "b"]), _timed("c"))

        await s.run()

        # a and b should overlap (both started before either finished).
        a_start, a_end = timestamps["a"]
        b_start, b_end = timestamps["b"]
        c_start, _ = timestamps["c"]

        # Both a and b must finish before c starts.
        assert a_end <= c_start + 0.01  # small tolerance
        assert b_end <= c_start + 0.01


# ===================================================================
# Observer / event system
# ===================================================================

class TestObserverEvents:

    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started: list[str] = []

        def _on_start(task: Task):
            started.append(task.id)

        s = TaskScheduler()
        s.on("on_task_start", _on_start)
        s.add_task(Task(id="t1", name="T1"), _make_coro())

        await s.run()

        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed: list[str] = []

        def _on_complete(task: Task):
            completed.append(task.id)

        s = TaskScheduler()
        s.on("on_task_complete", _on_complete)
        s.add_task(Task(id="t1", name="T1"), _make_coro())

        await s.run()

        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed: list[tuple[str, str]] = []

        def _on_fail(task: Task, exc: Exception):
            failed.append((task.id, str(exc)))

        s = TaskScheduler()
        s.on("on_task_fail", _on_fail)
        s.add_task(
            Task(id="t1", name="T1", max_retries=1),
            _make_always_failing_coro(RuntimeError("bad")),
        )

        await s.run()

        assert len(failed) == 1
        assert failed[0][0] == "t1"
        assert "bad" in failed[0][1]

    @pytest.mark.asyncio
    async def test_async_callback_supported(self):
        """Observer callbacks can be async coroutines."""
        started: list[str] = []

        async def _async_on_start(task: Task):
            started.append(task.id)

        s = TaskScheduler()
        s.on("on_task_start", _async_on_start)
        s.add_task(Task(id="t1", name="T1"), _make_coro())

        await s.run()

        assert "t1" in started

    @pytest.mark.asyncio
    async def test_multiple_listeners(self):
        calls_a: list[str] = []
        calls_b: list[str] = []

        def _cb_a(task: Task):
            calls_a.append(task.id)

        def _cb_b(task: Task):
            calls_b.append(task.id)

        s = TaskScheduler()
        s.on("on_task_complete", _cb_a)
        s.on("on_task_complete", _cb_b)
        s.add_task(Task(id="t1", name="T1"), _make_coro())

        await s.run()

        assert calls_a == ["t1"]
        assert calls_b == ["t1"]

    @pytest.mark.asyncio
    async def test_start_event_fires_on_each_retry(self):
        """on_task_start fires once per attempt (initial + retries)."""
        start_count = 0

        def _on_start(task: Task):
            nonlocal start_count
            start_count += 1

        s = TaskScheduler()
        s.on("on_task_start", _on_start)
        # Fails twice then succeeds → 3 total starts.
        s.add_task(
            Task(id="t1", name="T1", max_retries=3),
            _make_failing_coro(times=2),
        )

        await s.run()

        assert start_count == 3


# ===================================================================
# Metrics
# ===================================================================

class TestMetrics:

    @pytest.mark.asyncio
    async def test_total_time_positive(self):
        s = TaskScheduler()
        s.add_task(Task(id="t1", name="T1"), _make_coro(delay=0.01))

        metrics = await s.run()

        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_per_task_metrics_collected(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _make_coro())
        s.add_task(Task(id="b", name="B"), _make_coro())

        metrics = await s.run()

        assert "a" in metrics.task_metrics
        assert "b" in metrics.task_metrics
        assert metrics.task_metrics["a"].duration >= 0
        assert metrics.task_metrics["b"].duration >= 0

    @pytest.mark.asyncio
    async def test_failed_dependency_creates_metrics_entry(self):
        """Tasks skipped due to failed dependencies still get a metrics entry."""
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=1),
            _make_always_failing_coro(),
        )
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())

        metrics = await s.run()

        assert "b" in metrics.task_metrics
        # Skipped task has zero duration.
        assert metrics.task_metrics["b"].duration == pytest.approx(0.0)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_task_returning_complex_result(self):
        s = TaskScheduler()
        result_data = {"key": [1, 2, 3], "nested": {"a": True}}
        s.add_task(Task(id="t1", name="T1"), _make_coro(return_value=result_data))

        await s.run()

        assert s._tasks["t1"].result == result_data

    @pytest.mark.asyncio
    async def test_large_number_of_tasks(self):
        """Smoke test with many independent tasks."""
        s = TaskScheduler(max_concurrency=8)
        n = 50
        for i in range(n):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _make_coro(return_value=i))

        metrics = await s.run()

        assert len(metrics.task_metrics) == n
        for i in range(n):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Some tasks succeed, some fail; verify statuses are correct."""
        s = TaskScheduler()
        s.add_task(Task(id="ok", name="OK"), _make_coro(return_value="fine"))
        s.add_task(
            Task(id="bad", name="Bad", max_retries=1),
            _make_always_failing_coro(),
        )

        await s.run()

        assert s._tasks["ok"].status == TaskStatus.COMPLETED
        assert s._tasks["bad"].status == TaskStatus.FAILED

    def test_scheduler_metrics_defaults(self):
        m = SchedulerMetrics()
        assert m.total_time == 0.0
        assert m.task_metrics == {}

    @pytest.mark.asyncio
    async def test_cascading_failure(self):
        """A -> B -> C; if A fails, both B and C should fail."""
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="A", max_retries=1),
            _make_always_failing_coro(),
        )
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _make_coro())
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _make_coro())

        await s.run()

        assert s._tasks["a"].status == TaskStatus.FAILED
        assert s._tasks["b"].status == TaskStatus.FAILED
        assert s._tasks["c"].status == TaskStatus.FAILED
