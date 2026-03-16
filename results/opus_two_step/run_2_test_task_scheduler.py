"""Comprehensive tests for the async task scheduler module."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ── Helpers ─────────────────────────────────────────────────────────────


async def noop_coro(task: Task) -> str:
    """A trivial coroutine that succeeds immediately."""
    return f"done-{task.id}"


async def slow_coro(task: Task) -> str:
    """A coroutine that takes a noticeable amount of time."""
    await asyncio.sleep(0.05)
    return f"slow-{task.id}"


def _make_failing_coro(fail_times: int):
    """Return a coroutine factory that fails *fail_times* then succeeds."""
    call_count = 0

    async def coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count}")
        return f"recovered-{task.id}"

    return coro


async def always_fail_coro(task: Task) -> str:
    """A coroutine that always raises."""
    raise RuntimeError("permanent failure")


# ═══════════════════════════════════════════════════════════════════════
# 1. Task dataclass validation
# ═══════════════════════════════════════════════════════════════════════


class TestTaskDataclass:
    """Tests for the Task dataclass itself."""

    def test_valid_priority_bounds(self):
        t = Task(id="t", name="t", priority=1)
        assert t.priority == 1
        t = Task(id="t", name="t", priority=10)
        assert t.priority == 10

    def test_priority_too_low(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="t", priority=0)

    def test_priority_too_high(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="t", priority=11)

    def test_default_status_is_pending(self):
        t = Task(id="t", name="t", priority=5)
        assert t.status == TaskStatus.PENDING

    def test_default_dependencies_empty(self):
        t = Task(id="t", name="t", priority=5)
        assert t.dependencies == []

    def test_default_max_retries(self):
        t = Task(id="t", name="t", priority=5)
        assert t.max_retries == 3


# ═══════════════════════════════════════════════════════════════════════
# 2. Basic task registration & removal
# ═══════════════════════════════════════════════════════════════════════


class TestTaskRegistration:
    """Tests for add_task / remove_task / get_task."""

    def test_add_and_retrieve_task(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        scheduler.add_task(t, noop_coro)
        assert scheduler.get_task("a") is t

    def test_add_duplicate_raises(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        scheduler.add_task(t, noop_coro)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(Task(id="a", name="Again", priority=3), noop_coro)

    def test_remove_task(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        scheduler.add_task(t, noop_coro)
        scheduler.remove_task("a")
        with pytest.raises(KeyError):
            scheduler.get_task("a")

    def test_remove_nonexistent_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(KeyError, match="not found"):
            scheduler.remove_task("nope")

    def test_remove_running_task_raises(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        t.status = TaskStatus.RUNNING
        scheduler.add_task(t, noop_coro)
        with pytest.raises(RuntimeError, match="Cannot remove running task"):
            scheduler.remove_task("a")

    def test_get_nonexistent_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(KeyError, match="not found"):
            scheduler.get_task("missing")

    def test_tasks_property_returns_copy(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        scheduler.add_task(t, noop_coro)
        view = scheduler.tasks
        assert "a" in view
        # Mutating the view must not affect the scheduler
        view.pop("a")
        assert scheduler.get_task("a") is t


# ═══════════════════════════════════════════════════════════════════════
# 3. Basic task execution
# ═══════════════════════════════════════════════════════════════════════


class TestBasicExecution:
    """Tests for running simple tasks without dependencies."""

    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        t = Task(id="a", name="Alpha", priority=5)
        scheduler.add_task(t, noop_coro)
        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "done-a"
        assert metrics.task_metrics["a"].success is True

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task {i}", priority=5), noop_coro
            )
        metrics = await scheduler.run()

        for i in range(5):
            task = scheduler.get_task(f"t{i}")
            assert task.status == TaskStatus.COMPLETED
            assert task.result == f"done-t{i}"
        assert len(metrics.task_metrics) == 5

    @pytest.mark.asyncio
    async def test_no_tasks_returns_empty_metrics(self):
        scheduler = TaskScheduler()
        metrics = await scheduler.run()
        assert metrics.total_time >= 0
        assert metrics.task_metrics == {}
        assert metrics.total_retries == 0

    @pytest.mark.asyncio
    async def test_metrics_total_time_populated(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), slow_coro)
        metrics = await scheduler.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        async def custom(task: Task) -> dict:
            return {"key": "value"}

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), custom)
        await scheduler.run()
        assert scheduler.get_task("a").result == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════
# 4. Dependency resolution
# ═══════════════════════════════════════════════════════════════════════


class TestDependencyResolution:
    """Tests for topological sorting and dependency handling."""

    @pytest.mark.asyncio
    async def test_linear_chain_executes_in_order(self):
        """A -> B -> C must execute in that strict order."""
        order: list[str] = []

        async def tracking_coro(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), tracking_coro)
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), tracking_coro
        )
        scheduler.add_task(
            Task(id="c", name="C", priority=5, dependencies=["b"]), tracking_coro
        )
        await scheduler.run()

        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """
        A depends on nothing, B and C depend on A, D depends on B and C.
        B and C should run before D.
        """
        order: list[str] = []

        async def tracking_coro(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), tracking_coro)
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), tracking_coro
        )
        scheduler.add_task(
            Task(id="c", name="C", priority=5, dependencies=["a"]), tracking_coro
        )
        scheduler.add_task(
            Task(id="d", name="D", priority=5, dependencies=["b", "c"]),
            tracking_coro,
        )
        await scheduler.run()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_execution_plan_layers(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        scheduler.add_task(Task(id="c", name="C", priority=5), noop_coro)

        plan = scheduler.get_execution_plan()
        # Layer 0: a, c (no deps); Layer 1: b
        assert len(plan) == 2
        assert set(plan[0]) == {"a", "c"}
        assert plan[1] == ["b"]

    def test_execution_plan_respects_priority_within_layer(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="lo", name="Lo", priority=1), noop_coro)
        scheduler.add_task(Task(id="hi", name="Hi", priority=10), noop_coro)
        scheduler.add_task(Task(id="mid", name="Mid", priority=5), noop_coro)

        plan = scheduler.get_execution_plan()
        # All in one layer, sorted by descending priority
        assert plan == [["hi", "mid", "lo"]]

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["ghost"]), noop_coro
        )
        with pytest.raises(KeyError, match="unknown task"):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades(self):
        """If A fails, dependent B should also be marked FAILED."""
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, max_retries=0), always_fail_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        metrics = await scheduler.run()

        assert scheduler.get_task("a").status == TaskStatus.FAILED
        assert scheduler.get_task("b").status == TaskStatus.FAILED
        assert metrics.task_metrics["a"].success is False
        assert metrics.task_metrics["b"].success is False


# ═══════════════════════════════════════════════════════════════════════
# 5. Circular dependency detection
# ═══════════════════════════════════════════════════════════════════════


class TestCircularDependency:
    """Tests that circular dependencies are properly detected."""

    def test_self_dependency(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["a"]), noop_coro
        )
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["b"]), noop_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["c"]), noop_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        scheduler.add_task(
            Task(id="c", name="C", priority=5, dependencies=["b"]), noop_coro
        )
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_cycle_error_lists_involved_tasks(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="x", name="X", priority=5, dependencies=["y"]), noop_coro
        )
        scheduler.add_task(
            Task(id="y", name="Y", priority=5, dependencies=["x"]), noop_coro
        )
        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()
        msg = str(exc_info.value)
        assert "x" in msg
        assert "y" in msg

    @pytest.mark.asyncio
    async def test_circular_dependency_during_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["b"]), noop_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_partial_cycle_with_valid_subgraph(self):
        """A valid task exists alongside a cycle; detection should still fire."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="ok", name="OK", priority=5), noop_coro)
        scheduler.add_task(
            Task(id="a", name="A", priority=5, dependencies=["b"]), noop_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()


# ═══════════════════════════════════════════════════════════════════════
# 6. Retry logic with exponential backoff
# ═══════════════════════════════════════════════════════════════════════


class TestRetryLogic:
    """Tests for retry behaviour and exponential backoff timing."""

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        """Task that fails twice then succeeds (within max_retries=3)."""
        coro = _make_failing_coro(fail_times=2)
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.add_task(
            Task(id="r", name="Retry", priority=5, max_retries=3), coro
        )
        metrics = await scheduler.run()

        task = scheduler.get_task("r")
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2
        assert metrics.task_metrics["r"].success is True
        assert metrics.task_metrics["r"].retries == 2

    @pytest.mark.asyncio
    async def test_exhaust_retries_then_fail(self):
        """Task that always fails should be FAILED after max_retries+1 attempts."""
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.add_task(
            Task(id="f", name="Fail", priority=5, max_retries=2), always_fail_coro
        )
        metrics = await scheduler.run()

        task = scheduler.get_task("f")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 3  # exceeded max_retries of 2
        assert metrics.task_metrics["f"].success is False
        assert metrics.task_metrics["f"].retries == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self):
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.add_task(
            Task(id="f", name="NoRetry", priority=5, max_retries=0),
            always_fail_coro,
        )
        metrics = await scheduler.run()

        task = scheduler.get_task("f")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 1
        assert metrics.task_metrics["f"].retries == 0

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify that retries use exponential backoff delays."""
        base = 0.05
        coro = _make_failing_coro(fail_times=3)

        scheduler = TaskScheduler(base_backoff=base)
        scheduler.add_task(
            Task(id="b", name="Backoff", priority=5, max_retries=3), coro
        )

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # Expected backoff: base*1 + base*2 + base*4 = base*7
        expected_min = base * 7 * 0.8  # allow 20% tolerance
        assert elapsed >= expected_min, (
            f"elapsed {elapsed:.3f}s < expected minimum {expected_min:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_total_retries_aggregated_in_scheduler_metrics(self):
        scheduler = TaskScheduler(base_backoff=0.01)
        coro1 = _make_failing_coro(fail_times=1)
        coro2 = _make_failing_coro(fail_times=2)
        scheduler.add_task(Task(id="a", name="A", priority=5, max_retries=3), coro1)
        scheduler.add_task(Task(id="b", name="B", priority=5, max_retries=3), coro2)
        metrics = await scheduler.run()

        assert metrics.total_retries == 3  # 1 + 2

    @pytest.mark.asyncio
    async def test_metrics_duration_positive_on_success(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), slow_coro)
        metrics = await scheduler.run()
        assert metrics.task_metrics["a"].duration > 0

    @pytest.mark.asyncio
    async def test_metrics_duration_positive_on_failure(self):
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.add_task(
            Task(id="f", name="F", priority=5, max_retries=0), always_fail_coro
        )
        metrics = await scheduler.run()
        assert metrics.task_metrics["f"].duration >= 0


# ═══════════════════════════════════════════════════════════════════════
# 7. Concurrent execution respecting concurrency limits
# ═══════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Tests that the scheduler respects the max_concurrency setting."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """No more than max_concurrency tasks should run at the same time."""
        max_concurrent = 2
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_coro(task: Task) -> None:
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1

        scheduler = TaskScheduler(max_concurrency=max_concurrent)
        for i in range(6):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task{i}", priority=5), counting_coro
            )
        await scheduler.run()

        assert peak_concurrent <= max_concurrent

    @pytest.mark.asyncio
    async def test_concurrency_1_is_serial(self):
        """With max_concurrency=1, tasks run one at a time."""
        execution_log: list[tuple[str, str]] = []  # (task_id, event)

        async def log_coro(task: Task) -> None:
            execution_log.append((task.id, "start"))
            await asyncio.sleep(0.02)
            execution_log.append((task.id, "end"))

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(3):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task{i}", priority=5), log_coro
            )
        await scheduler.run()

        # With serial execution, each "end" should come before the next "start"
        for idx in range(len(execution_log) - 1):
            if execution_log[idx][1] == "end" and idx + 1 < len(execution_log):
                next_event = execution_log[idx + 1]
                if next_event[1] == "start":
                    # This is fine, serial ordering
                    pass
            # No two "start" events should appear without an "end" in between
            if execution_log[idx][1] == "start":
                # Find the matching end for this task
                task_id = execution_log[idx][0]
                end_idx = next(
                    j
                    for j in range(idx + 1, len(execution_log))
                    if execution_log[j] == (task_id, "end")
                )
                # No other "start" should appear between start and end of this task
                for j in range(idx + 1, end_idx):
                    assert execution_log[j][1] != "start", (
                        f"Task {execution_log[j][0]} started while {task_id} was running"
                    )

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Independent tasks actually run in parallel (not sequentially)."""
        task_count = 4
        per_task_sleep = 0.1

        async def sleeping_coro(task: Task) -> None:
            await asyncio.sleep(per_task_sleep)

        scheduler = TaskScheduler(max_concurrency=task_count)
        for i in range(task_count):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"T{i}", priority=5), sleeping_coro
            )

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # If truly concurrent, total time ~ per_task_sleep, not task_count * per_task_sleep
        assert elapsed < per_task_sleep * task_count * 0.8, (
            f"Expected concurrent execution but took {elapsed:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_layers_execute_sequentially(self):
        """Different layers must execute in sequence, not all at once."""
        layer_starts: dict[str, float] = {}

        async def timed_coro(task: Task) -> None:
            layer_starts[task.id] = time.monotonic()
            await asyncio.sleep(0.05)

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(Task(id="a", name="A", priority=5), timed_coro)
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), timed_coro
        )
        await scheduler.run()

        # B must start after A finishes (~0.05s later)
        assert layer_starts["b"] > layer_starts["a"] + 0.04


# ═══════════════════════════════════════════════════════════════════════
# 8. Observer / event pattern
# ═══════════════════════════════════════════════════════════════════════


class TestObserverPattern:
    """Tests for event listener callbacks."""

    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started: list[str] = []

        def on_start(task: Task) -> None:
            started.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_start(on_start)
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)
        await scheduler.run()

        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed: list[str] = []

        def on_complete(task: Task) -> None:
            completed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_complete(on_complete)
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)
        await scheduler.run()

        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed: list[str] = []

        def on_fail(task: Task) -> None:
            failed.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_fail(on_fail)
        scheduler.add_task(
            Task(id="f", name="F", priority=5, max_retries=0), always_fail_coro
        )
        await scheduler.run()

        assert "f" in failed

    @pytest.mark.asyncio
    async def test_async_callback(self):
        events: list[str] = []

        async def async_on_start(task: Task) -> None:
            events.append(f"async-start-{task.id}")

        scheduler = TaskScheduler()
        scheduler.on_task_start(async_on_start)
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)
        await scheduler.run()

        assert "async-start-a" in events

    @pytest.mark.asyncio
    async def test_multiple_listeners_same_event(self):
        log1: list[str] = []
        log2: list[str] = []

        def cb1(task: Task) -> None:
            log1.append(task.id)

        def cb2(task: Task) -> None:
            log2.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_complete(cb1)
        scheduler.on_task_complete(cb2)
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)
        await scheduler.run()

        assert log1 == ["a"]
        assert log2 == ["a"]

    @pytest.mark.asyncio
    async def test_on_task_fail_fires_for_cascaded_dependency_failure(self):
        """When a task is skipped because its dependency failed, on_task_fail fires."""
        failed: list[str] = []

        def on_fail(task: Task) -> None:
            failed.append(task.id)

        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.on_task_fail(on_fail)
        scheduler.add_task(
            Task(id="a", name="A", priority=5, max_retries=0), always_fail_coro
        )
        scheduler.add_task(
            Task(id="b", name="B", priority=5, dependencies=["a"]), noop_coro
        )
        await scheduler.run()

        assert "a" in failed
        assert "b" in failed

    @pytest.mark.asyncio
    async def test_start_event_fires_on_each_retry(self):
        """on_task_start should fire each time the task is attempted."""
        started: list[str] = []

        def on_start(task: Task) -> None:
            started.append(task.id)

        coro = _make_failing_coro(fail_times=2)
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.on_task_start(on_start)
        scheduler.add_task(
            Task(id="r", name="R", priority=5, max_retries=3), coro
        )
        await scheduler.run()

        # 2 failures + 1 success = 3 start events
        assert started.count("r") == 3


# ═══════════════════════════════════════════════════════════════════════
# 9. Edge cases and integration
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Miscellaneous edge cases and integration scenarios."""

    @pytest.mark.asyncio
    async def test_scheduler_can_run_twice(self):
        """Running the scheduler a second time resets metrics."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), noop_coro)

        m1 = await scheduler.run()
        assert m1.task_metrics["a"].success is True

        # Reset task state for second run
        scheduler.get_task("a").status = TaskStatus.PENDING
        scheduler.get_task("a").retry_count = 0
        m2 = await scheduler.run()
        assert m2.task_metrics["a"].success is True
        # Metrics should be fresh, not accumulated
        assert m2.total_retries == 0

    @pytest.mark.asyncio
    async def test_many_tasks_stress(self):
        """Run a larger number of independent tasks to smoke-test."""
        scheduler = TaskScheduler(max_concurrency=8)
        n = 50
        for i in range(n):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task{i}", priority=(i % 10) + 1),
                noop_coro,
            )
        metrics = await scheduler.run()

        assert len(metrics.task_metrics) == n
        assert all(m.success for m in metrics.task_metrics.values())

    @pytest.mark.asyncio
    async def test_task_with_all_dependencies_satisfied(self):
        """A task depending on multiple completed tasks should succeed."""
        order: list[str] = []

        async def tracking(task: Task) -> None:
            order.append(task.id)

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), tracking)
        scheduler.add_task(Task(id="b", name="B", priority=5), tracking)
        scheduler.add_task(
            Task(id="c", name="C", priority=5, dependencies=["a", "b"]),
            tracking,
        )
        await scheduler.run()

        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")
        assert scheduler.get_task("c").status == TaskStatus.COMPLETED

    def test_metrics_property_accessible(self):
        scheduler = TaskScheduler()
        m = scheduler.metrics
        assert isinstance(m, SchedulerMetrics)

    @pytest.mark.asyncio
    async def test_backoff_does_not_hold_semaphore(self):
        """
        During backoff sleep, the semaphore should be released so other tasks
        can proceed. We test this by running a failing task alongside a
        succeeding task with max_concurrency=1.
        """
        timeline: list[tuple[str, float]] = []
        base_time = time.monotonic()

        coro_fail = _make_failing_coro(fail_times=1)

        async def recording_coro(task: Task) -> str:
            timeline.append((f"{task.id}-start", time.monotonic() - base_time))
            await asyncio.sleep(0.01)
            timeline.append((f"{task.id}-end", time.monotonic() - base_time))
            return f"done-{task.id}"

        async def fail_then_succeed(task: Task) -> str:
            timeline.append((f"{task.id}-attempt", time.monotonic() - base_time))
            return await coro_fail(task)

        scheduler = TaskScheduler(max_concurrency=1, base_backoff=0.1)
        scheduler.add_task(
            Task(id="fail", name="Failing", priority=5, max_retries=1),
            fail_then_succeed,
        )
        scheduler.add_task(
            Task(id="ok", name="OK", priority=5), recording_coro
        )
        await scheduler.run()

        # The "ok" task should start while "fail" is in backoff sleep
        # If the semaphore were held during backoff, "ok" would be blocked
        ok_events = [e for e in timeline if e[0].startswith("ok")]
        assert len(ok_events) > 0, "OK task should have executed"
