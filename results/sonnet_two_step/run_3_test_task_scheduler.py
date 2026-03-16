"""
test_task_scheduler.py

Comprehensive pytest test suite for task_scheduler.py.

Coverage areas:
  - Basic task execution
  - Dependency resolution
  - Circular dependency detection
  - Retry logic with exponential backoff
  - Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    SchedulerMetrics,
    Task,
    TaskNotFoundError,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(
    id_: str,
    name: str = "",
    priority: int = 5,
    dependencies: list[str] | None = None,
    max_retries: int = 0,
) -> Task:
    return Task(
        id=id_,
        name=name or id_,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def noop() -> str:
    return "ok"


async def failing() -> None:
    raise ValueError("intentional failure")


# ---------------------------------------------------------------------------
# Task dataclass validation
# ---------------------------------------------------------------------------

class TestTaskDataclass:
    def test_valid_priority_boundaries(self):
        t1 = make_task("t", priority=1)
        t10 = make_task("t2", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_zero(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="t", name="t", priority=0)

    def test_invalid_priority_eleven(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="t", name="t", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_zero(self):
        t = make_task("t")
        assert t.retry_count == 0


# ---------------------------------------------------------------------------
# Scheduler registration
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_single_task(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        plan = sched.get_execution_plan()
        assert plan == [["a"]]

    def test_duplicate_task_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        with pytest.raises(ValueError, match="already registered"):
            sched.add_task(make_task("a"), noop)


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        metrics = await sched.run()
        assert sched._tasks["a"].status == TaskStatus.COMPLETED
        assert sched._tasks["a"].result == "ok"
        assert isinstance(metrics, SchedulerMetrics)

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        async def produce() -> int:
            return 42

        sched = TaskScheduler()
        sched.add_task(make_task("a"), produce)
        await sched.run()
        assert sched._tasks["a"].result == 42

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        sched = TaskScheduler()
        for tid in ("a", "b", "c"):
            sched.add_task(make_task(tid), noop)
        await sched.run()
        for tid in ("a", "b", "c"):
            assert sched._tasks[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_metrics_elapsed_seconds_positive(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        metrics = await sched.run()
        assert metrics.total_elapsed_seconds >= 0
        assert metrics.task_metrics["a"].elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_task_status_running_during_execution(self):
        statuses_seen: list[TaskStatus] = []

        async def observe() -> None:
            # By the time the callable runs the status is already RUNNING
            # (set in _run_task before the callable is invoked).
            pass

        sched = TaskScheduler()
        task = make_task("a")
        start_events: list[TaskStatus] = []

        async def on_start(t: Task) -> None:
            start_events.append(t.status)

        sched.on_task_start(on_start)
        sched.add_task(task, observe)
        await sched.run()
        assert start_events == [TaskStatus.RUNNING]


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_linear_chain_plan(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        sched.add_task(make_task("b", dependencies=["a"]), noop)
        sched.add_task(make_task("c", dependencies=["b"]), noop)
        plan = sched.get_execution_plan()
        assert plan == [["a"], ["b"], ["c"]]

    def test_diamond_dependency_plan(self):
        # a -> b, a -> c, b -> d, c -> d
        sched = TaskScheduler()
        sched.add_task(make_task("a"), noop)
        sched.add_task(make_task("b", dependencies=["a"]), noop)
        sched.add_task(make_task("c", dependencies=["a"]), noop)
        sched.add_task(make_task("d", dependencies=["b", "c"]), noop)
        plan = sched.get_execution_plan()
        assert plan[0] == ["a"]
        # b and c may appear in either order within the second layer
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    @pytest.mark.asyncio
    async def test_dependency_executes_before_dependent(self):
        order: list[str] = []

        async def record(tid: str):
            async def _inner():
                order.append(tid)
            return _inner

        sched = TaskScheduler()
        sched.add_task(make_task("a"), await record("a"))
        sched.add_task(make_task("b", dependencies=["a"]), await record("b"))
        await sched.run()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_chain_of_three_respects_order(self):
        order: list[str] = []

        def make_recorder(tid: str):
            async def _inner():
                order.append(tid)
            return _inner

        sched = TaskScheduler()
        sched.add_task(make_task("a"), make_recorder("a"))
        sched.add_task(make_task("b", dependencies=["a"]), make_recorder("b"))
        sched.add_task(make_task("c", dependencies=["b"]), make_recorder("c"))
        await sched.run()
        assert order == ["a", "b", "c"]

    def test_unknown_dependency_raises_task_not_found(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["nonexistent"]), noop)
        with pytest.raises(TaskNotFoundError, match="nonexistent"):
            sched.get_execution_plan()

    def test_unknown_dependency_raises_on_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["ghost"]), noop)
        with pytest.raises(TaskNotFoundError):
            asyncio.get_event_loop().run_until_complete(sched.run())

    @pytest.mark.asyncio
    async def test_failed_dependency_cascades_to_dependent(self):
        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=0), failing)
        sched.add_task(make_task("b", dependencies=["a"], max_retries=0), noop)

        with pytest.raises(RuntimeError):
            await sched.run()

        assert sched._tasks["b"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_independent_task_runs_despite_sibling_failure(self):
        """A task not depending on a failed peer should still complete."""
        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=0), failing)
        sched.add_task(make_task("b"), noop)  # independent

        with pytest.raises(RuntimeError):
            await sched.run()

        assert sched._tasks["b"].status == TaskStatus.COMPLETED

    def test_priority_ordering_within_layer(self):
        sched = TaskScheduler()
        sched.add_task(make_task("low", priority=1), noop)
        sched.add_task(make_task("mid", priority=5), noop)
        sched.add_task(make_task("high", priority=9), noop)
        plan = sched.get_execution_plan()
        # All three are in the same layer; highest priority appears first.
        assert plan[0][0] == "high"
        assert plan[0][-1] == "low"


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_simple_two_node_cycle(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]), noop)
        sched.add_task(make_task("b", dependencies=["a"]), noop)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_three_node_cycle(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["c"]), noop)
        sched.add_task(make_task("b", dependencies=["a"]), noop)
        sched.add_task(make_task("c", dependencies=["b"]), noop)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_self_loop(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["a"]), noop)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_cycle_in_subset_raises(self):
        """Acyclic tasks exist alongside a cycle; cycle is still detected."""
        sched = TaskScheduler()
        sched.add_task(make_task("ok1"), noop)
        sched.add_task(make_task("ok2", dependencies=["ok1"]), noop)
        sched.add_task(make_task("x", dependencies=["y"]), noop)
        sched.add_task(make_task("y", dependencies=["x"]), noop)
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_error_message_contains_cycle_participants(self):
        sched = TaskScheduler()
        sched.add_task(make_task("alpha", dependencies=["beta"]), noop)
        sched.add_task(make_task("beta", dependencies=["alpha"]), noop)
        with pytest.raises(CircularDependencyError) as exc_info:
            sched.get_execution_plan()
        msg = str(exc_info.value)
        assert "alpha" in msg or "beta" in msg

    @pytest.mark.asyncio
    async def test_run_raises_circular_dependency(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]), noop)
        sched.add_task(make_task("b", dependencies=["a"]), noop)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    def test_no_false_positive_for_valid_dag(self):
        sched = TaskScheduler()
        # Build a more complex valid DAG
        sched.add_task(make_task("a"), noop)
        sched.add_task(make_task("b"), noop)
        sched.add_task(make_task("c", dependencies=["a"]), noop)
        sched.add_task(make_task("d", dependencies=["a", "b"]), noop)
        sched.add_task(make_task("e", dependencies=["c", "d"]), noop)
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
                raise ValueError("not yet")
            return "done"

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=3), flaky)
        await sched.run()
        assert call_count == 3
        assert sched._tasks["a"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=2), failing)
        with pytest.raises(RuntimeError, match="a"):
            await sched.run()
        assert sched._tasks["a"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_count_recorded_on_task(self):
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=3), always_fails)
        with pytest.raises(RuntimeError):
            await sched.run()
        # retry_count increments once per failed attempt before the last one
        assert sched._tasks["a"].retry_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_recorded_in_metrics(self):
        async def always_fails():
            raise RuntimeError("boom")

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=2), always_fails)
        with pytest.raises(RuntimeError):
            await sched.run()
        assert sched.metrics.task_metrics["a"].retry_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.add_task(make_task("a", max_retries=5), succeed)
        await sched.run()
        assert call_count == 1
        assert sched._tasks["a"].retry_count == 0

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleep_durations(self):
        """Verify asyncio.sleep is called with exponentially increasing delays."""
        sleep_calls: list[float] = []

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_fails():
            raise RuntimeError("boom")

        sched = TaskScheduler(base_backoff_seconds=1.0)
        sched.add_task(make_task("a", max_retries=3), always_fails)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await sched.run()

        # 3 retries → delays 1*2^0=1, 1*2^1=2, 1*2^2=4
        assert sleep_calls == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_base_backoff_scaling(self):
        """base_backoff_seconds parameter scales all sleep durations."""
        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_fails():
            raise RuntimeError("boom")

        sched = TaskScheduler(base_backoff_seconds=0.5)
        sched.add_task(make_task("a", max_retries=2), always_fails)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await sched.run()

        assert sleep_calls == [0.5, 1.0]

    @pytest.mark.asyncio
    async def test_on_task_fail_event_emitted(self):
        failed_ids: list[str] = []

        async def on_fail(task: Task) -> None:
            failed_ids.append(task.id)

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.on_task_fail(on_fail)
        sched.add_task(make_task("a", max_retries=0), failing)

        with pytest.raises(RuntimeError):
            await sched.run()

        assert "a" in failed_ids

    @pytest.mark.asyncio
    async def test_eventual_success_does_not_emit_fail_event(self):
        failed_ids: list[str] = []

        async def on_fail(task: Task) -> None:
            failed_ids.append(task.id)

        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("once")
            return "ok"

        sched = TaskScheduler(base_backoff_seconds=0)
        sched.on_task_fail(on_fail)
        sched.add_task(make_task("a", max_retries=3), flaky)
        await sched.run()
        assert "a" not in failed_ids


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_max_workers_one_serializes_execution(self):
        """With max_workers=1 tasks run one at a time even in the same layer."""
        active_count = 0
        max_active = 0

        async def track():
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0)  # yield to event loop
            active_count -= 1

        sched = TaskScheduler(max_workers=1)
        for i in range(5):
            sched.add_task(make_task(f"t{i}"), track)

        await sched.run()
        assert max_active == 1

    @pytest.mark.asyncio
    async def test_max_workers_respected_under_load(self):
        limit = 3
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def track():
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            async with lock:
                active_count -= 1

        sched = TaskScheduler(max_workers=limit)
        for i in range(10):
            sched.add_task(make_task(f"t{i}"), track)

        await sched.run()
        assert max_active <= limit

    @pytest.mark.asyncio
    async def test_concurrent_tasks_complete_faster_than_serial(self):
        """Multiple workers should finish parallel tasks faster than serial."""

        async def slow():
            await asyncio.sleep(0.05)

        # Serial baseline: max_workers=1, 4 tasks → ~0.20 s
        sched_serial = TaskScheduler(max_workers=1)
        for i in range(4):
            sched_serial.add_task(make_task(f"t{i}"), slow)
        t0 = time.monotonic()
        await sched_serial.run()
        serial_time = time.monotonic() - t0

        # Concurrent: max_workers=4, same 4 tasks → ~0.05 s
        sched_parallel = TaskScheduler(max_workers=4)
        for i in range(4):
            sched_parallel.add_task(make_task(f"t{i}"), slow)
        t0 = time.monotonic()
        await sched_parallel.run()
        parallel_time = time.monotonic() - t0

        assert parallel_time < serial_time * 0.75

    @pytest.mark.asyncio
    async def test_default_max_workers_is_four(self):
        """Default scheduler allows up to 4 concurrent tasks."""
        active_count = 0
        peak = 0
        lock = asyncio.Lock()

        async def slow():
            nonlocal active_count, peak
            async with lock:
                active_count += 1
                peak = max(peak, active_count)
            await asyncio.sleep(0.05)
            async with lock:
                active_count -= 1

        sched = TaskScheduler()  # default max_workers=4
        for i in range(8):
            sched.add_task(make_task(f"t{i}"), slow)

        await sched.run()
        assert peak <= 4

    @pytest.mark.asyncio
    async def test_semaphore_not_exceeded_with_dependencies(self):
        """Dependency layers + concurrency limit must both be respected."""
        limit = 2
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def track():
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            async with lock:
                active_count -= 1

        sched = TaskScheduler(max_workers=limit)
        # Layer 1: a, b, c (independent)
        sched.add_task(make_task("a"), track)
        sched.add_task(make_task("b"), track)
        sched.add_task(make_task("c"), track)
        # Layer 2: d depends on a, e depends on b
        sched.add_task(make_task("d", dependencies=["a"]), track)
        sched.add_task(make_task("e", dependencies=["b"]), track)

        await sched.run()
        assert max_active <= limit


# ---------------------------------------------------------------------------
# Observer / event emission
# ---------------------------------------------------------------------------

class TestEventEmission:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        started: list[str] = []

        async def handler(task: Task) -> None:
            started.append(task.id)

        sched = TaskScheduler()
        sched.on_task_start(handler)
        sched.add_task(make_task("a"), noop)
        await sched.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        completed: list[str] = []

        async def handler(task: Task) -> None:
            completed.append(task.id)

        sched = TaskScheduler()
        sched.on_task_complete(handler)
        sched.add_task(make_task("a"), noop)
        await sched.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_called(self):
        log: list[str] = []

        async def h1(task: Task) -> None:
            log.append("h1")

        async def h2(task: Task) -> None:
            log.append("h2")

        sched = TaskScheduler()
        sched.on_task_complete(h1)
        sched.on_task_complete(h2)
        sched.add_task(make_task("a"), noop)
        await sched.run()
        assert "h1" in log and "h2" in log

    @pytest.mark.asyncio
    async def test_event_order_start_before_complete(self):
        events: list[str] = []

        async def on_start(task: Task) -> None:
            events.append(f"start:{task.id}")

        async def on_complete(task: Task) -> None:
            events.append(f"complete:{task.id}")

        sched = TaskScheduler()
        sched.on_task_start(on_start)
        sched.on_task_complete(on_complete)
        sched.add_task(make_task("a"), noop)
        await sched.run()
        assert events.index("start:a") < events.index("complete:a")
