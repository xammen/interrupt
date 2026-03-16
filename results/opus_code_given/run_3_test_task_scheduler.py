"""Comprehensive tests for the async task scheduler module."""

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


# ── Helpers ──────────────────────────────────────────────────────────────────


async def noop_coro(task: Task) -> str:
    """A trivial coroutine that returns a fixed string."""
    return f"done-{task.id}"


async def slow_coro(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate work."""
    await asyncio.sleep(0.05)
    return f"slow-{task.id}"


def _make_failing_coro(fail_times: int):
    """Return a coroutine factory that fails the first *fail_times* calls."""
    call_count = 0

    async def _coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count}")
        return f"recovered-{task.id}"

    return _coro


# ═══════════════════════════════════════════════════════════════════════════
# 1. Task & TaskMetrics dataclass basics
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskDataclass:
    def test_defaults(self):
        t = Task(id="t1", name="Test")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None
        assert t.created_at is not None

    def test_priority_boundaries(self):
        Task(id="lo", name="Lo", priority=1)
        Task(id="hi", name="Hi", priority=10)

    def test_priority_too_low(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="X", priority=0)

    def test_priority_too_high(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="x", name="X", priority=11)


class TestTaskMetricsDataclass:
    def test_defaults(self):
        m = TaskMetrics()
        assert m.total_time == 0.0
        assert m.per_task_time == {}
        assert m.retry_counts == {}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scheduler construction & task management
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerConstruction:
    def test_default_concurrency(self):
        s = TaskScheduler()
        assert s.max_concurrency == 4

    def test_custom_concurrency(self):
        s = TaskScheduler(max_concurrency=2)
        assert s.max_concurrency == 2

    def test_invalid_concurrency(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)

    def test_add_and_get_task(self):
        s = TaskScheduler()
        t = Task(id="a", name="A")
        s.add_task(t, noop_coro)
        assert s.get_task("a") is t

    def test_add_duplicate_task_raises(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="A2"), noop_coro)

    def test_get_missing_task_raises(self):
        s = TaskScheduler()
        with pytest.raises(KeyError):
            s.get_task("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Basic task execution
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        metrics = await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "done-a"
        assert "a" in metrics.per_task_time
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        s = TaskScheduler()
        for letter in "abc":
            s.add_task(Task(id=letter, name=letter.upper()), noop_coro)

        metrics = await s.run()

        for letter in "abc":
            assert s.get_task(letter).status == TaskStatus.COMPLETED
            assert s.get_task(letter).result == f"done-{letter}"
        assert len(metrics.per_task_time) == 3

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        async def return_42(task: Task) -> int:
            return 42

        s = TaskScheduler()
        s.add_task(Task(id="x", name="X"), return_42)
        await s.run()
        assert s.get_task("x").result == 42

    @pytest.mark.asyncio
    async def test_metrics_retry_counts_zero_on_success(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        metrics = await s.run()
        assert metrics.retry_counts["a"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dependency resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestDependencyResolution:
    def test_linear_chain_execution_plan(self):
        """a -> b -> c should produce three sequential groups."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), noop_coro)

        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_diamond_dependency(self):
        """
        a -> b
        a -> c
        b,c -> d
        Should give 3 groups: [a], [b,c], [d].
        """
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), noop_coro)

        plan = s.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    def test_independent_tasks_in_same_group(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X"), noop_coro)
        s.add_task(Task(id="y", name="Y"), noop_coro)

        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"x", "y"}

    def test_priority_ordering_within_group(self):
        """Higher-priority tasks should appear first within a concurrent group."""
        s = TaskScheduler()
        s.add_task(Task(id="lo", name="Low", priority=1), noop_coro)
        s.add_task(Task(id="hi", name="High", priority=10), noop_coro)

        plan = s.get_execution_plan()
        assert plan[0][0] == "hi"
        assert plan[0][1] == "lo"

    def test_unknown_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["missing"]), noop_coro)

        with pytest.raises(KeyError, match="unknown task 'missing'"):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_dependencies_run_in_order(self):
        """Track execution order to verify dependencies are respected."""
        order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            order.append(task.id)
            return task.id

        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="first", name="First"), tracking_coro)
        s.add_task(Task(id="second", name="Second", dependencies=["first"]), tracking_coro)
        s.add_task(Task(id="third", name="Third", dependencies=["second"]), tracking_coro)

        await s.run()
        assert order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades(self):
        """When a dependency fails, dependent tasks should also be marked FAILED."""
        async def always_fail(task: Task) -> None:
            raise RuntimeError("boom")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), always_fail)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)

        await s.run()
        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED
        # b was never actually run, so it has no result
        assert s.get_task("b").result is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Circular dependency detection
# ═══════════════════════════════════════════════════════════════════════════


class TestCircularDependencyDetection:
    def test_self_dependency(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["a"]), noop_coro)

        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_two_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)

        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert exc_info.value.cycle  # cycle should be non-empty

    def test_three_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), noop_coro)

        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        # All three should appear in the cycle
        assert len(exc_info.value.cycle) >= 2

    def test_cycle_among_subset_with_valid_tasks(self):
        """Only some tasks form a cycle; others are fine."""
        s = TaskScheduler()
        s.add_task(Task(id="ok", name="OK"), noop_coro)
        s.add_task(Task(id="a", name="A", dependencies=["b"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)

        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_circular_dependency_during_run(self):
        """run() should also raise CircularDependencyError."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)

        with pytest.raises(CircularDependencyError):
            await s.run()

    def test_error_message_includes_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X", dependencies=["y"]), noop_coro)
        s.add_task(Task(id="y", name="Y", dependencies=["x"]), noop_coro)

        with pytest.raises(CircularDependencyError, match="Circular dependency detected"):
            s.get_execution_plan()

    def test_circular_dependency_error_without_cycle(self):
        err = CircularDependencyError()
        assert err.cycle == []
        assert "Circular dependency detected" in str(err)

    def test_circular_dependency_error_with_cycle(self):
        err = CircularDependencyError(cycle=["a", "b", "a"])
        assert err.cycle == ["a", "b", "a"]
        assert "a -> b -> a" in str(err)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Retry logic with exponential backoff
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_then_succeeds(self):
        """Task fails twice then succeeds on 3rd attempt (within max_retries=3)."""
        coro = _make_failing_coro(fail_times=2)

        s = TaskScheduler()
        s.add_task(Task(id="r", name="Retry", max_retries=3), coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            metrics = await s.run()

        assert s.get_task("r").status == TaskStatus.COMPLETED
        assert s.get_task("r").result == "recovered-r"
        assert metrics.retry_counts["r"] == 2
        # Two retries => two backoff sleeps: 2^0=1, 2^1=2
        assert mock_sleep.await_count == 2
        mock_sleep.assert_any_await(1)
        mock_sleep.assert_any_await(2)

    @pytest.mark.asyncio
    async def test_task_exhausts_retries_and_fails(self):
        """Task always fails and should be marked FAILED after max_retries."""
        async def always_fail(task: Task) -> None:
            raise RuntimeError("fail forever")

        s = TaskScheduler()
        s.add_task(Task(id="f", name="Fail", max_retries=2), always_fail)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            metrics = await s.run()

        assert s.get_task("f").status == TaskStatus.FAILED
        # retry_count increments each failure: 1, 2, 3 (the 3rd exceeds max_retries=2)
        assert s.get_task("f").retry_count == 3
        assert metrics.retry_counts["f"] == 3

    @pytest.mark.asyncio
    async def test_zero_max_retries(self):
        """With max_retries=0, the first failure immediately marks FAILED."""
        async def fail_once(task: Task) -> None:
            raise RuntimeError("one shot")

        s = TaskScheduler()
        s.add_task(Task(id="z", name="Zero", max_retries=0), fail_once)

        metrics = await s.run()
        assert s.get_task("z").status == TaskStatus.FAILED
        assert s.get_task("z").retry_count == 1
        assert metrics.retry_counts["z"] == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_values(self):
        """Verify the exact backoff durations: 1, 2, 4 seconds for retries 1-3."""
        async def always_fail(task: Task) -> None:
            raise RuntimeError("fail")

        s = TaskScheduler()
        s.add_task(Task(id="b", name="Backoff", max_retries=3), always_fail)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await s.run()

        # 3 retries before exceeding max_retries → 3 backoff sleeps
        expected_backoffs = [1, 2, 4]  # 2^0, 2^1, 2^2
        actual_backoffs = [call.args[0] for call in mock_sleep.await_args_list]
        assert actual_backoffs == expected_backoffs

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self):
        coro = _make_failing_coro(fail_times=1)

        s = TaskScheduler()
        s.add_task(Task(id="m", name="M", max_retries=3), coro)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            metrics = await s.run()

        assert metrics.retry_counts["m"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. Concurrent execution respecting concurrency limits
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Ensure no more than max_concurrency tasks run simultaneously."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency(task: Task) -> str:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return task.id

        limit = 2
        s = TaskScheduler(max_concurrency=limit)
        for i in range(6):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), track_concurrency)

        await s.run()
        assert max_concurrent <= limit

    @pytest.mark.asyncio
    async def test_concurrency_1_is_serial(self):
        """With max_concurrency=1, tasks should run one at a time."""
        order: list[str] = []

        async def record(task: Task) -> str:
            order.append(f"start-{task.id}")
            await asyncio.sleep(0.01)
            order.append(f"end-{task.id}")
            return task.id

        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="a", name="A"), record)
        s.add_task(Task(id="b", name="B"), record)

        await s.run()
        # With concurrency=1 and both in same group, each should fully
        # complete before the next starts OR overlap—but semaphore ensures
        # at most 1 holds the semaphore at a time.
        # The exact interleaving depends on the event loop, but we verify
        # start/end pairs don't interleave.
        for task_id in ["a", "b"]:
            start_idx = order.index(f"start-{task_id}")
            end_idx = order.index(f"end-{task_id}")
            assert end_idx == start_idx + 1

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_concurrency(self):
        """All tasks should complete even with limited concurrency."""
        s = TaskScheduler(max_concurrency=2)
        num_tasks = 10
        for i in range(num_tasks):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), noop_coro)

        metrics = await s.run()

        for i in range(num_tasks):
            assert s.get_task(f"t{i}").status == TaskStatus.COMPLETED
        assert len(metrics.per_task_time) == num_tasks

    @pytest.mark.asyncio
    async def test_groups_execute_sequentially(self):
        """Tasks in later groups must wait for earlier groups to finish."""
        timestamps: dict[str, tuple[float, float]] = {}

        async def timed_coro(task: Task) -> str:
            start = time.monotonic()
            await asyncio.sleep(0.05)
            end = time.monotonic()
            timestamps[task.id] = (start, end)
            return task.id

        s = TaskScheduler(max_concurrency=4)
        s.add_task(Task(id="a", name="A"), timed_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), timed_coro)

        await s.run()

        # b should not start before a finishes
        assert timestamps["b"][0] >= timestamps["a"][1] - 0.01  # small tolerance


# ═══════════════════════════════════════════════════════════════════════════
# 8. Observer pattern (event listeners)
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverPattern:
    @pytest.mark.asyncio
    async def test_on_task_start_event(self):
        started: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.on("on_task_start", lambda t: started.append(t.id))

        await s.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_event(self):
        completed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.on("on_task_complete", lambda t: completed.append(t.id))

        await s.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_event(self):
        failed: list[str] = []

        async def always_fail(task: Task) -> None:
            raise RuntimeError("boom")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), always_fail)
        s.on("on_task_fail", lambda t: failed.append(t.id))

        await s.run()
        assert "a" in failed

    @pytest.mark.asyncio
    async def test_async_listener(self):
        events: list[str] = []

        async def async_callback(task: Task) -> None:
            events.append(f"async-{task.id}")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.on("on_task_complete", async_callback)

        await s.run()
        assert "async-a" in events

    @pytest.mark.asyncio
    async def test_multiple_listeners_for_same_event(self):
        log1: list[str] = []
        log2: list[str] = []

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.on("on_task_complete", lambda t: log1.append(t.id))
        s.on("on_task_complete", lambda t: log2.append(t.id))

        await s.run()
        assert "a" in log1
        assert "a" in log2

    @pytest.mark.asyncio
    async def test_event_order_start_before_complete(self):
        events: list[str] = []

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.on("on_task_start", lambda t: events.append(f"start-{t.id}"))
        s.on("on_task_complete", lambda t: events.append(f"complete-{t.id}"))

        await s.run()
        assert events.index("start-a") < events.index("complete-a")

    @pytest.mark.asyncio
    async def test_fail_event_for_cascaded_dependency_failure(self):
        """When a task fails due to dependency failure, on_task_fail should fire."""
        failed: list[str] = []

        async def always_fail(task: Task) -> None:
            raise RuntimeError("boom")

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), always_fail)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.on("on_task_fail", lambda t: failed.append(t.id))

        await s.run()
        assert "a" in failed
        assert "b" in failed


# ═══════════════════════════════════════════════════════════════════════════
# 9. Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestMetrics:
    @pytest.mark.asyncio
    async def test_total_time_is_positive(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        metrics = await s.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_per_task_time_recorded(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), slow_coro)
        metrics = await s.run()
        assert metrics.per_task_time["a"] >= 0.04  # slept ~0.05s

    @pytest.mark.asyncio
    async def test_metrics_reset_between_runs(self):
        """Each call to run() should produce fresh metrics."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)

        m1 = await s.run()
        # Reset statuses for re-run (tasks reuse same objects)
        s.get_task("a").status = TaskStatus.PENDING
        s.get_task("a").retry_count = 0

        m2 = await s.run()
        # Both should have metrics, and they should be different objects
        assert m1 is not m2
        assert "a" in m2.per_task_time


# ═══════════════════════════════════════════════════════════════════════════
# 10. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_fine(self):
        s = TaskScheduler()
        metrics = await s.run()
        assert metrics.total_time >= 0
        assert metrics.per_task_time == {}

    @pytest.mark.asyncio
    async def test_single_task_no_dependencies(self):
        s = TaskScheduler()
        s.add_task(Task(id="solo", name="Solo"), noop_coro)
        await s.run()
        assert s.get_task("solo").status == TaskStatus.COMPLETED

    def test_execution_plan_empty_scheduler(self):
        s = TaskScheduler()
        plan = s.get_execution_plan()
        assert plan == []

    @pytest.mark.asyncio
    async def test_many_tasks_large_dependency_graph(self):
        """Stress test: chain of 50 dependent tasks."""
        s = TaskScheduler()
        for i in range(50):
            deps = [f"t{i-1}"] if i > 0 else []
            s.add_task(Task(id=f"t{i}", name=f"T{i}", dependencies=deps), noop_coro)

        metrics = await s.run()
        for i in range(50):
            assert s.get_task(f"t{i}").status == TaskStatus.COMPLETED
        assert len(metrics.per_task_time) == 50

    @pytest.mark.asyncio
    async def test_wide_fan_out(self):
        """One root task with many dependents, all in one group."""
        s = TaskScheduler(max_concurrency=4)
        s.add_task(Task(id="root", name="Root"), noop_coro)
        for i in range(20):
            s.add_task(
                Task(id=f"leaf{i}", name=f"Leaf{i}", dependencies=["root"]),
                noop_coro,
            )

        plan = s.get_execution_plan()
        assert len(plan) == 2
        assert plan[0] == ["root"]
        assert len(plan[1]) == 20

        metrics = await s.run()
        assert s.get_task("root").status == TaskStatus.COMPLETED
        for i in range(20):
            assert s.get_task(f"leaf{i}").status == TaskStatus.COMPLETED
