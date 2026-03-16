"""Comprehensive tests for task_scheduler module."""

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


async def _noop(task: Task) -> str:
    """Trivial coroutine that returns a fixed value."""
    return f"result-{task.id}"


async def _slow(task: Task) -> str:
    """Coroutine that sleeps briefly to simulate real work."""
    await asyncio.sleep(0.05)
    return f"slow-{task.id}"


def _make_failing_coro(fail_times: int):
    """Return an async callable that raises on the first *fail_times* calls."""
    call_count = 0

    async def _coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"transient failure #{call_count}")
        return f"recovered-{task.id}"

    return _coro


# ===================================================================
# Task dataclass
# ===================================================================


class TestTaskDataclass:
    """Tests for the Task dataclass itself."""

    def test_default_values(self):
        t = Task(id="t1", name="Task 1")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_lower_bound(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=0)

    def test_priority_upper_bound(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=11)

    def test_priority_boundaries_valid(self):
        t_low = Task(id="t1", name="low", priority=1)
        t_high = Task(id="t2", name="high", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10


# ===================================================================
# Metrics dataclasses
# ===================================================================


class TestMetrics:
    def test_task_metrics_duration_none_when_incomplete(self):
        m = TaskMetrics(task_id="t1", start_time=1.0)
        assert m.duration is None

    def test_task_metrics_duration_calculated(self):
        m = TaskMetrics(task_id="t1", start_time=1.0, end_time=3.5)
        assert m.duration == pytest.approx(2.5)

    def test_scheduler_metrics_total_time_none_initially(self):
        m = SchedulerMetrics()
        assert m.total_time is None

    def test_scheduler_metrics_total_time_calculated(self):
        m = SchedulerMetrics(total_start=10.0, total_end=15.0)
        assert m.total_time == pytest.approx(5.0)


# ===================================================================
# Task management (add / get)
# ===================================================================


class TestTaskManagement:
    def test_add_and_get_task(self):
        scheduler = TaskScheduler()
        task = Task(id="t1", name="Task 1")
        scheduler.add_task(task, _noop)
        assert scheduler.get_task("t1") is task

    def test_add_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Task 1"), _noop)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(Task(id="t1", name="Dup"), _noop)

    def test_get_nonexistent_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            scheduler.get_task("missing")


# ===================================================================
# Basic task execution
# ===================================================================


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Only"), _noop)

        results = await scheduler.run()

        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t1"].result == "result-t1"

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_complete(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(Task(id=f"t{i}", name=f"Task {i}"), _noop)

        results = await scheduler.run()

        for i in range(5):
            assert results[f"t{i}"].status == TaskStatus.COMPLETED
            assert results[f"t{i}"].result == f"result-t{i}"

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        async def custom(task: Task) -> dict:
            return {"key": "value", "id": task.id}

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Custom"), custom)
        results = await scheduler.run()

        assert results["t1"].result == {"key": "value", "id": "t1"}

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Metered"), _noop)

        await scheduler.run()

        m = scheduler.metrics
        assert m.total_time is not None
        assert m.total_time >= 0
        assert "t1" in m.task_metrics
        assert m.task_metrics["t1"].duration is not None
        assert m.task_metrics["t1"].duration >= 0

    @pytest.mark.asyncio
    async def test_event_on_task_start_fires(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Ev"), _noop)

        started: list[str] = []
        scheduler.on("on_task_start", lambda t: started.append(t.id))

        await scheduler.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_event_on_task_complete_fires(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="Ev"), _noop)

        completed: list[str] = []
        scheduler.on("on_task_complete", lambda t: completed.append(t.id))

        await scheduler.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_async_event_callback(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="AsyncCb"), _noop)

        log: list[str] = []

        async def async_cb(task: Task) -> None:
            log.append(f"async-{task.id}")

        scheduler.on("on_task_complete", async_cb)
        await scheduler.run()

        assert "async-t1" in log


# ===================================================================
# Dependency resolution
# ===================================================================


class TestDependencyResolution:
    def test_linear_chain_execution_plan(self):
        """A -> B -> C produces three sequential groups."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A"), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)
        scheduler.add_task(Task(id="C", name="C", dependencies=["B"]), _noop)

        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["A"]
        assert plan[1] == ["B"]
        assert plan[2] == ["C"]

    def test_diamond_dependency_plan(self):
        """Diamond: A -> B, A -> C, B+C -> D."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A"), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)
        scheduler.add_task(Task(id="C", name="C", dependencies=["A"]), _noop)
        scheduler.add_task(
            Task(id="D", name="D", dependencies=["B", "C"]), _noop
        )

        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["A"]
        assert set(plan[1]) == {"B", "C"}
        assert plan[2] == ["D"]

    def test_independent_tasks_single_group(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A"), _noop)
        scheduler.add_task(Task(id="B", name="B"), _noop)
        scheduler.add_task(Task(id="C", name="C"), _noop)

        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"A", "B", "C"}

    def test_priority_ordering_within_group(self):
        """Within a single group, higher-priority tasks appear first."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="low", name="Low", priority=1), _noop)
        scheduler.add_task(Task(id="high", name="High", priority=10), _noop)
        scheduler.add_task(Task(id="med", name="Med", priority=5), _noop)

        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert plan[0] == ["high", "med", "low"]

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="A", name="A", dependencies=["nonexistent"]), _noop
        )
        with pytest.raises(TaskNotFoundError, match="unknown task"):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_dependency_order_respected_at_runtime(self):
        """Ensure dependent tasks actually run after their prerequisites."""
        execution_order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            execution_order.append(task.id)
            return task.id

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A"), tracking_coro)
        scheduler.add_task(
            Task(id="B", name="B", dependencies=["A"]), tracking_coro
        )
        scheduler.add_task(
            Task(id="C", name="C", dependencies=["B"]), tracking_coro
        )

        await scheduler.run()

        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("B") < execution_order.index("C")

    @pytest.mark.asyncio
    async def test_cascade_failure_marks_downstream(self):
        """If task A fails, dependent B and C should also be FAILED."""

        async def always_fail(task: Task) -> None:
            raise RuntimeError("boom")

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", max_retries=0), always_fail)
        scheduler.add_task(
            Task(id="B", name="B", dependencies=["A"]), _noop
        )
        scheduler.add_task(
            Task(id="C", name="C", dependencies=["B"]), _noop
        )

        results = await scheduler.run()

        assert results["A"].status == TaskStatus.FAILED
        assert results["B"].status == TaskStatus.FAILED
        assert results["C"].status == TaskStatus.FAILED
        # B and C should have RuntimeError results about dependency failure
        assert isinstance(results["B"].result, RuntimeError)
        assert "Dependency" in str(results["B"].result)

    @pytest.mark.asyncio
    async def test_partial_failure_independent_tasks_still_run(self):
        """A failure in one branch does not affect independent tasks."""

        async def fail(task: Task) -> None:
            raise RuntimeError("fail")

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", max_retries=0), fail)
        scheduler.add_task(Task(id="B", name="B"), _noop)  # independent

        results = await scheduler.run()

        assert results["A"].status == TaskStatus.FAILED
        assert results["B"].status == TaskStatus.COMPLETED


# ===================================================================
# Circular dependency detection
# ===================================================================


class TestCircularDependency:
    def test_simple_cycle_detected(self):
        """A -> B -> A should raise."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", dependencies=["B"]), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)

        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()

        assert len(exc_info.value.cycle) >= 2
        # The cycle attribute should contain the involved nodes
        assert "A" in exc_info.value.cycle
        assert "B" in exc_info.value.cycle

    def test_self_cycle_detected(self):
        """A task depending on itself should raise."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", dependencies=["A"]), _noop)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self):
        """A -> B -> C -> A should raise."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", dependencies=["C"]), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)
        scheduler.add_task(Task(id="C", name="C", dependencies=["B"]), _noop)

        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()

        cycle = exc_info.value.cycle
        # Cycle should start and end with the same node
        assert cycle[0] == cycle[-1]

    def test_cycle_in_subgraph_detected(self):
        """Cycle exists only in a subgraph; independent task D is fine."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", dependencies=["B"]), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)
        scheduler.add_task(Task(id="D", name="D"), _noop)  # independent

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_cycle_prevents_run(self):
        """scheduler.run() should raise on circular deps before executing."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A", dependencies=["B"]), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)

        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_no_false_positive_on_dag(self):
        """A valid DAG (diamond shape) should NOT raise."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="A", name="A"), _noop)
        scheduler.add_task(Task(id="B", name="B", dependencies=["A"]), _noop)
        scheduler.add_task(Task(id="C", name="C", dependencies=["A"]), _noop)
        scheduler.add_task(
            Task(id="D", name="D", dependencies=["B", "C"]), _noop
        )

        plan = scheduler.get_execution_plan()  # should not raise
        assert len(plan) == 3

    def test_error_message_contains_cycle_path(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="X", name="X", dependencies=["Y"]), _noop)
        scheduler.add_task(Task(id="Y", name="Y", dependencies=["X"]), _noop)

        with pytest.raises(CircularDependencyError, match="Circular dependency detected"):
            scheduler.get_execution_plan()


# ===================================================================
# Retry logic with exponential backoff
# ===================================================================


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_then_succeeds(self):
        """Fail twice, succeed on third attempt (retry_count=2)."""
        coro = _make_failing_coro(fail_times=2)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="Flaky", max_retries=3), coro
        )

        results = await scheduler.run()

        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t1"].result == "recovered-t1"
        assert results["t1"].retry_count == 2

    @pytest.mark.asyncio
    async def test_task_exhausts_retries_and_fails(self):
        """Fail more times than max_retries allows."""
        coro = _make_failing_coro(fail_times=100)  # always fails

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="Doomed", max_retries=2), coro
        )

        results = await scheduler.run()

        assert results["t1"].status == TaskStatus.FAILED
        assert isinstance(results["t1"].result, RuntimeError)
        # max_retries=2 means: initial attempt + 2 retries = 3 total attempts
        # retry_count should be 3 (attempt > max_retries triggers failure at attempt=3)
        assert results["t1"].retry_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        """With max_retries=0, the task fails on the first exception."""
        coro = _make_failing_coro(fail_times=1)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="NoRetry", max_retries=0), coro
        )

        results = await scheduler.run()

        assert results["t1"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify that retries respect exponential backoff delays.

        Backoff formula: 2^(attempt-1) * 0.1
          attempt 1 -> 0.1s
          attempt 2 -> 0.2s
        Total minimum delay ~ 0.3s for 2 retries before succeeding on 3rd try.
        """
        coro = _make_failing_coro(fail_times=2)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="Backoff", max_retries=3), coro
        )

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # 0.1 + 0.2 = 0.3s minimum; give generous margin
        assert elapsed >= 0.25, f"Expected >= 0.25s, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_on_task_fail_event_on_exhausted_retries(self):
        """on_task_fail fires when retries are exhausted."""
        coro = _make_failing_coro(fail_times=100)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="FailEv", max_retries=1), coro
        )

        fail_log: list[tuple[str, str]] = []
        scheduler.on(
            "on_task_fail",
            lambda t, exc: fail_log.append((t.id, str(exc))),
        )

        await scheduler.run()

        assert len(fail_log) == 1
        assert fail_log[0][0] == "t1"

    @pytest.mark.asyncio
    async def test_on_task_start_fires_each_retry(self):
        """on_task_start should fire once per attempt (initial + retries)."""
        coro = _make_failing_coro(fail_times=2)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="StartEv", max_retries=3), coro
        )

        start_count: list[str] = []
        scheduler.on("on_task_start", lambda t: start_count.append(t.id))

        await scheduler.run()

        # 2 failures + 1 success = 3 starts
        assert len(start_count) == 3

    @pytest.mark.asyncio
    async def test_metrics_record_retries(self):
        coro = _make_failing_coro(fail_times=2)

        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="t1", name="MetricRetry", max_retries=3), coro
        )

        await scheduler.run()

        task_metric = scheduler.metrics.task_metrics["t1"]
        assert task_metric.retries == 2


# ===================================================================
# Concurrent execution respecting concurrency limits
# ===================================================================


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At most max_concurrency tasks should run simultaneously."""
        max_concurrency = 2
        peak = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked(task: Task) -> str:
            nonlocal peak, current
            async with lock:
                current += 1
                if current > peak:
                    peak = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return task.id

        scheduler = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            scheduler.add_task(Task(id=f"t{i}", name=f"T{i}"), tracked)

        await scheduler.run()

        assert peak <= max_concurrency

    @pytest.mark.asyncio
    async def test_concurrency_limit_one_serialises(self):
        """With max_concurrency=1, tasks must run one at a time."""
        execution_log: list[tuple[str, str]] = []

        async def logged(task: Task) -> str:
            execution_log.append((task.id, "start"))
            await asyncio.sleep(0.02)
            execution_log.append((task.id, "end"))
            return task.id

        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(3):
            scheduler.add_task(Task(id=f"t{i}", name=f"T{i}"), logged)

        await scheduler.run()

        # Verify no interleaving: pattern must be start/end/start/end/…
        for i in range(0, len(execution_log), 2):
            assert execution_log[i][1] == "start"
            assert execution_log[i + 1][1] == "end"
            assert execution_log[i][0] == execution_log[i + 1][0]

    @pytest.mark.asyncio
    async def test_tasks_within_group_run_concurrently(self):
        """Independent tasks in the same group should actually overlap."""
        timestamps: dict[str, tuple[float, float]] = {}

        async def timed(task: Task) -> str:
            start = time.monotonic()
            await asyncio.sleep(0.1)
            end = time.monotonic()
            timestamps[task.id] = (start, end)
            return task.id

        scheduler = TaskScheduler(max_concurrency=4)
        for i in range(4):
            scheduler.add_task(Task(id=f"t{i}", name=f"T{i}"), timed)

        await scheduler.run()

        # With concurrency=4 and 4 tasks, total wall time should be near
        # 0.1s, not 0.4s (serial). Allow generous margin.
        total = scheduler.metrics.total_time
        assert total is not None
        assert total < 0.35, f"Expected concurrent execution, got {total:.3f}s"

    @pytest.mark.asyncio
    async def test_groups_run_sequentially(self):
        """Tasks in group 2 must start AFTER all tasks in group 1 finish."""
        timestamps: dict[str, tuple[float, float]] = {}

        async def timed(task: Task) -> str:
            start = time.monotonic()
            await asyncio.sleep(0.05)
            end = time.monotonic()
            timestamps[task.id] = (start, end)
            return task.id

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(Task(id="A", name="A"), timed)
        scheduler.add_task(
            Task(id="B", name="B", dependencies=["A"]), timed
        )

        await scheduler.run()

        a_end = timestamps["A"][1]
        b_start = timestamps["B"][0]
        assert b_start >= a_end, "B should start after A finishes"

    @pytest.mark.asyncio
    async def test_large_fan_out_respects_limit(self):
        """Many tasks depending on a single root, limited concurrency."""
        max_conc = 3
        peak = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked(task: Task) -> str:
            nonlocal peak, current
            async with lock:
                current += 1
                if current > peak:
                    peak = current
            await asyncio.sleep(0.03)
            async with lock:
                current -= 1
            return task.id

        scheduler = TaskScheduler(max_concurrency=max_conc)
        scheduler.add_task(Task(id="root", name="Root"), tracked)
        for i in range(10):
            scheduler.add_task(
                Task(id=f"leaf{i}", name=f"Leaf{i}", dependencies=["root"]),
                tracked,
            )

        await scheduler.run()

        assert peak <= max_conc

    @pytest.mark.asyncio
    async def test_all_tasks_complete_under_concurrency(self):
        """Every registered task should reach COMPLETED even with a tight limit."""
        scheduler = TaskScheduler(max_concurrency=2)
        ids = [f"t{i}" for i in range(8)]
        for tid in ids:
            scheduler.add_task(Task(id=tid, name=tid), _noop)

        results = await scheduler.run()

        for tid in ids:
            assert results[tid].status == TaskStatus.COMPLETED


# ===================================================================
# Edge cases / integration
# ===================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_run(self):
        """Running with no tasks should succeed and return empty dict."""
        scheduler = TaskScheduler()
        results = await scheduler.run()
        assert results == {}

    @pytest.mark.asyncio
    async def test_single_task_no_deps(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="only", name="Only"), _noop)
        results = await scheduler.run()
        assert results["only"].status == TaskStatus.COMPLETED

    def test_execution_plan_empty_scheduler(self):
        scheduler = TaskScheduler()
        plan = scheduler.get_execution_plan()
        assert plan == []

    @pytest.mark.asyncio
    async def test_metrics_reset_between_runs(self):
        """Metrics should be reset on each call to run()."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t1", name="T"), _noop)

        await scheduler.run()
        first_start = scheduler.metrics.total_start

        # Reset task statuses for a second run
        scheduler._tasks["t1"].status = TaskStatus.PENDING

        await scheduler.run()
        second_start = scheduler.metrics.total_start

        assert second_start is not None
        assert first_start is not None
        assert second_start > first_start

    @pytest.mark.asyncio
    async def test_cascade_failure_deep_chain(self):
        """Failure at the root of a long chain cascades to all descendants."""

        async def fail(task: Task) -> None:
            raise RuntimeError("root fails")

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="t0", name="Root", max_retries=0), fail)
        for i in range(1, 5):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"T{i}", dependencies=[f"t{i-1}"]),
                _noop,
            )

        results = await scheduler.run()

        for i in range(5):
            assert results[f"t{i}"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_concurrent_retries_do_not_deadlock(self):
        """Multiple failing tasks retrying concurrently should not deadlock."""
        scheduler = TaskScheduler(max_concurrency=2)
        for i in range(4):
            coro = _make_failing_coro(fail_times=1)  # fail once, then succeed
            scheduler.add_task(
                Task(id=f"t{i}", name=f"T{i}", max_retries=2), coro
            )

        results = await scheduler.run()

        for i in range(4):
            assert results[f"t{i}"].status == TaskStatus.COMPLETED
