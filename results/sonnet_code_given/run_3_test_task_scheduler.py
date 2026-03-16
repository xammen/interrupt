"""
test_task_scheduler.py - Comprehensive pytest test suite for task_scheduler.py

Covers:
- Basic task execution
- Dependency resolution
- Circular dependency detection
- Retry logic with exponential backoff
- Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    EventBus,
    SchedulerMetrics,
    Task,
    TaskMetrics,
    TaskNotFoundError,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_task(
    task_id: str,
    name: str = "task",
    priority: int = 5,
    dependencies: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def succeeding_handler(task: Task) -> str:
    return f"result_of_{task.id}"


async def failing_handler(task: Task) -> None:
    raise ValueError(f"forced failure in {task.id}")


@pytest.fixture
def scheduler() -> TaskScheduler:
    return TaskScheduler(max_concurrency=4, base_backoff=0.0)


@pytest.fixture
def fast_scheduler() -> TaskScheduler:
    """Scheduler with zero backoff for retry tests so they don't slow the suite."""
    return TaskScheduler(max_concurrency=4, base_backoff=0.0)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_boundaries(self):
        t_low = make_task("a", priority=1)
        t_high = make_task("b", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            make_task("a", priority=0)

        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            make_task("b", priority=11)

    def test_default_status_is_pending(self):
        assert make_task("a").status == TaskStatus.PENDING

    def test_default_retry_count_is_zero(self):
        assert make_task("a").retry_count == 0

    def test_default_result_is_none(self):
        assert make_task("a").result is None

    def test_default_dependencies_is_empty_list(self):
        assert make_task("a").dependencies == []

    def test_created_at_is_set(self):
        t = make_task("a")
        assert t.created_at is not None


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_when_not_started(self):
        m = TaskMetrics(task_id="a")
        assert m.elapsed is None

    def test_elapsed_none_when_only_start_set(self):
        m = TaskMetrics(task_id="a", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed_correctly(self):
        m = TaskMetrics(task_id="a", start_time=1.0, end_time=4.0)
        assert m.elapsed == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_time_none_when_not_finished(self):
        sm = SchedulerMetrics()
        assert sm.total_time is None

    def test_total_time_computed(self):
        sm = SchedulerMetrics(total_start=0.0, total_end=5.0)
        assert sm.total_time == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_calls_registered_listener(self):
        bus = EventBus()
        received: list[Task] = []

        async def cb(task: Task) -> None:
            received.append(task)

        bus.subscribe("my_event", cb)
        t = make_task("x")
        await bus.emit("my_event", t)
        assert received == [t]

    @pytest.mark.asyncio
    async def test_emit_calls_multiple_listeners(self):
        bus = EventBus()
        calls: list[str] = []

        async def cb1(task: Task) -> None:
            calls.append("cb1")

        async def cb2(task: Task) -> None:
            calls.append("cb2")

        bus.subscribe("evt", cb1)
        bus.subscribe("evt", cb2)
        await bus.emit("evt", make_task("y"))
        assert calls == ["cb1", "cb2"]

    @pytest.mark.asyncio
    async def test_emit_unknown_event_does_nothing(self):
        bus = EventBus()
        # Should not raise
        await bus.emit("nonexistent_event", make_task("z"))

    @pytest.mark.asyncio
    async def test_subscribe_different_events_isolated(self):
        bus = EventBus()
        fired: list[str] = []

        async def cb_a(task: Task) -> None:
            fired.append("a")

        async def cb_b(task: Task) -> None:
            fired.append("b")

        bus.subscribe("event_a", cb_a)
        bus.subscribe("event_b", cb_b)

        await bus.emit("event_a", make_task("t"))
        assert fired == ["a"]

        await bus.emit("event_b", make_task("t"))
        assert fired == ["a", "b"]


# ---------------------------------------------------------------------------
# TaskScheduler – registration
# ---------------------------------------------------------------------------

class TestTaskSchedulerRegistration:
    def test_add_task_succeeds(self, scheduler):
        scheduler.add_task(make_task("t1"), succeeding_handler)
        assert "t1" in scheduler._tasks

    def test_add_duplicate_task_raises(self, scheduler):
        scheduler.add_task(make_task("t1"), succeeding_handler)
        with pytest.raises(ValueError, match="already registered"):
            scheduler.add_task(make_task("t1"), succeeding_handler)


# ---------------------------------------------------------------------------
# TaskScheduler – dependency validation
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    def test_unknown_dependency_raises_task_not_found(self, scheduler):
        scheduler.add_task(make_task("child", dependencies=["ghost"]), succeeding_handler)
        with pytest.raises(TaskNotFoundError, match="unknown task 'ghost'"):
            scheduler.get_execution_plan()

    def test_known_dependency_passes_validation(self, scheduler):
        scheduler.add_task(make_task("parent"), succeeding_handler)
        scheduler.add_task(make_task("child", dependencies=["parent"]), succeeding_handler)
        plan = scheduler.get_execution_plan()
        assert plan  # does not raise


# ---------------------------------------------------------------------------
# TaskScheduler – execution plan (topological sort)
# ---------------------------------------------------------------------------

class TestGetExecutionPlan:
    def test_single_task_plan(self, scheduler):
        scheduler.add_task(make_task("solo"), succeeding_handler)
        plan = scheduler.get_execution_plan()
        assert plan == [["solo"]]

    def test_independent_tasks_in_same_group(self, scheduler):
        for tid in ("a", "b", "c"):
            scheduler.add_task(make_task(tid), succeeding_handler)
        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b", "c"}

    def test_linear_chain_produces_sequential_groups(self, scheduler):
        scheduler.add_task(make_task("first"), succeeding_handler)
        scheduler.add_task(make_task("second", dependencies=["first"]), succeeding_handler)
        scheduler.add_task(make_task("third", dependencies=["second"]), succeeding_handler)
        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["first"]
        assert plan[1] == ["second"]
        assert plan[2] == ["third"]

    def test_diamond_dependency_plan(self, scheduler):
        """
        root → left, right → merge
        Expected: [[root], [left, right], [merge]]
        """
        scheduler.add_task(make_task("root"), succeeding_handler)
        scheduler.add_task(make_task("left", dependencies=["root"]), succeeding_handler)
        scheduler.add_task(make_task("right", dependencies=["root"]), succeeding_handler)
        scheduler.add_task(
            make_task("merge", dependencies=["left", "right"]), succeeding_handler
        )
        plan = scheduler.get_execution_plan()
        assert plan[0] == ["root"]
        assert set(plan[1]) == {"left", "right"}
        assert plan[2] == ["merge"]

    def test_within_group_sorted_by_descending_priority(self, scheduler):
        scheduler.add_task(make_task("low", priority=2), succeeding_handler)
        scheduler.add_task(make_task("high", priority=8), succeeding_handler)
        scheduler.add_task(make_task("mid", priority=5), succeeding_handler)
        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert plan[0] == ["high", "mid", "low"]

    def test_empty_scheduler_returns_empty_plan(self, scheduler):
        assert scheduler.get_execution_plan() == []


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_direct_cycle_raises(self, scheduler):
        scheduler.add_task(make_task("a", dependencies=["b"]), succeeding_handler)
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_handler)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle_raises(self, scheduler):
        scheduler.add_task(make_task("a", dependencies=["c"]), succeeding_handler)
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_handler)
        scheduler.add_task(make_task("c", dependencies=["b"]), succeeding_handler)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_cycle_error_message_contains_involved_tasks(self, scheduler):
        scheduler.add_task(make_task("x", dependencies=["y"]), succeeding_handler)
        scheduler.add_task(make_task("y", dependencies=["x"]), succeeding_handler)
        with pytest.raises(CircularDependencyError, match="x|y"):
            scheduler.get_execution_plan()

    def test_self_loop_raises(self, scheduler):
        # A task depending on itself
        scheduler.add_task(make_task("self_ref", dependencies=["self_ref"]), succeeding_handler)
        with pytest.raises((CircularDependencyError, TaskNotFoundError)):
            scheduler.get_execution_plan()

    def test_partial_cycle_with_valid_tasks_raises(self, scheduler):
        """Ensure detection even when some tasks are acyclic."""
        scheduler.add_task(make_task("ok"), succeeding_handler)
        scheduler.add_task(make_task("a", dependencies=["ok", "b"]), succeeding_handler)
        scheduler.add_task(make_task("b", dependencies=["a"]), succeeding_handler)
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self, fast_scheduler):
        task = make_task("t1")
        fast_scheduler.add_task(task, succeeding_handler)
        await fast_scheduler.run()
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self, fast_scheduler):
        task = make_task("t1")
        fast_scheduler.add_task(task, succeeding_handler)
        await fast_scheduler.run()
        assert task.result == "result_of_t1"

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self, fast_scheduler):
        tasks = [make_task(f"t{i}") for i in range(5)]
        for t in tasks:
            fast_scheduler.add_task(t, succeeding_handler)
        await fast_scheduler.run()
        for t in tasks:
            assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self, fast_scheduler):
        task = make_task("t1")
        fast_scheduler.add_task(task, succeeding_handler)
        metrics = await fast_scheduler.run()
        assert "t1" in metrics.per_task
        assert metrics.per_task["t1"].start_time is not None
        assert metrics.per_task["t1"].end_time is not None
        assert metrics.per_task["t1"].elapsed is not None
        assert metrics.per_task["t1"].elapsed >= 0

    @pytest.mark.asyncio
    async def test_scheduler_metrics_total_time_set(self, fast_scheduler):
        fast_scheduler.add_task(make_task("t1"), succeeding_handler)
        metrics = await fast_scheduler.run()
        assert metrics.total_time is not None
        assert metrics.total_time >= 0

    @pytest.mark.asyncio
    async def test_on_task_start_event_fired(self, fast_scheduler):
        started: list[str] = []

        async def cb(task: Task) -> None:
            started.append(task.id)

        fast_scheduler.on_task_start(cb)
        fast_scheduler.add_task(make_task("t1"), succeeding_handler)
        await fast_scheduler.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_event_fired(self, fast_scheduler):
        completed: list[str] = []

        async def cb(task: Task) -> None:
            completed.append(task.id)

        fast_scheduler.on_task_complete(cb)
        fast_scheduler.add_task(make_task("t1"), succeeding_handler)
        await fast_scheduler.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_handler_receives_task_object(self, fast_scheduler):
        received: list[Task] = []

        async def capturing_handler(task: Task) -> str:
            received.append(task)
            return "ok"

        t = make_task("t1")
        fast_scheduler.add_task(t, capturing_handler)
        await fast_scheduler.run()
        assert received[0] is t


# ---------------------------------------------------------------------------
# Dependency resolution during execution
# ---------------------------------------------------------------------------

class TestDependencyExecution:
    @pytest.mark.asyncio
    async def test_parent_completes_before_child_starts(self, fast_scheduler):
        order: list[str] = []

        async def tracking_handler(task: Task) -> str:
            order.append(task.id)
            return "ok"

        fast_scheduler.add_task(make_task("parent"), tracking_handler)
        fast_scheduler.add_task(
            make_task("child", dependencies=["parent"]), tracking_handler
        )
        await fast_scheduler.run()
        assert order.index("parent") < order.index("child")

    @pytest.mark.asyncio
    async def test_chain_executes_in_order(self, fast_scheduler):
        order: list[str] = []

        async def tracking_handler(task: Task) -> str:
            order.append(task.id)
            return "ok"

        fast_scheduler.add_task(make_task("a"), tracking_handler)
        fast_scheduler.add_task(make_task("b", dependencies=["a"]), tracking_handler)
        fast_scheduler.add_task(make_task("c", dependencies=["b"]), tracking_handler)
        await fast_scheduler.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_all_complete(self, fast_scheduler):
        fast_scheduler.add_task(make_task("root"), succeeding_handler)
        fast_scheduler.add_task(make_task("left", dependencies=["root"]), succeeding_handler)
        fast_scheduler.add_task(make_task("right", dependencies=["root"]), succeeding_handler)
        fast_scheduler.add_task(
            make_task("merge", dependencies=["left", "right"]), succeeding_handler
        )
        await fast_scheduler.run()
        for tid in ("root", "left", "right", "merge"):
            assert fast_scheduler._tasks[tid].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure(self, fast_scheduler):
        call_count = 0

        async def flaky_handler(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary error")
            return "ok"

        t = make_task("flaky", max_retries=3)
        fast_scheduler.add_task(t, flaky_handler)
        await fast_scheduler.run()
        assert t.status == TaskStatus.COMPLETED
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_incremented(self, fast_scheduler):
        async def always_fail(task: Task) -> None:
            raise RuntimeError("always fails")

        t = make_task("failing", max_retries=2)
        fast_scheduler.add_task(t, always_fail)
        with pytest.raises(RuntimeError):
            await fast_scheduler.run()
        assert t.retry_count == 3  # initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_task_status_failed_after_exhausting_retries(self, fast_scheduler):
        t = make_task("failing", max_retries=1)
        fast_scheduler.add_task(t, failing_handler)
        with pytest.raises(RuntimeError):
            await fast_scheduler.run()
        assert t.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_on_task_fail_event_fired(self, fast_scheduler):
        failed: list[str] = []

        async def cb(task: Task) -> None:
            failed.append(task.id)

        fast_scheduler.on_task_fail(cb)
        t = make_task("bad", max_retries=0)
        fast_scheduler.add_task(t, failing_handler)
        with pytest.raises(RuntimeError):
            await fast_scheduler.run()
        assert "bad" in failed

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self, fast_scheduler):
        call_count = 0

        async def counting_handler(task: Task) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        t = make_task("t", max_retries=0)
        fast_scheduler.add_task(t, counting_handler)
        with pytest.raises(RuntimeError):
            await fast_scheduler.run()
        assert call_count == 1  # only the initial attempt, zero retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays_increase(self):
        """Verify that asyncio.sleep is called with exponentially increasing delays."""
        sleep_calls: list[float] = []

        async def always_fail(task: Task) -> None:
            raise RuntimeError("fail")

        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            # Do NOT actually sleep to keep the test fast

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            s = TaskScheduler(max_concurrency=4, base_backoff=1.0)
            t = make_task("t", max_retries=3)
            s.add_task(t, always_fail)
            with pytest.raises(RuntimeError):
                await s.run()

        # Should have 3 sleep calls (after attempt 1, 2, 3) with 1, 2, 4 seconds
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)
        assert sleep_calls[2] == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_no_sleep_after_final_failed_attempt(self):
        """asyncio.sleep should NOT be called after the last retry is exhausted."""
        sleep_calls: list[float] = []

        async def always_fail(task: Task) -> None:
            raise RuntimeError("fail")

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("task_scheduler.asyncio.sleep", side_effect=mock_sleep):
            s = TaskScheduler(max_concurrency=4, base_backoff=1.0)
            t = make_task("t", max_retries=2)
            s.add_task(t, always_fail)
            with pytest.raises(RuntimeError):
                await s.run()

        # max_retries=2: sleep after attempt 1, sleep after attempt 2, NO sleep after attempt 3
        assert len(sleep_calls) == 2

    @pytest.mark.asyncio
    async def test_metrics_reflect_retry_count(self, fast_scheduler):
        call_count = 0

        async def flaky(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("retry me")
            return "ok"

        t = make_task("t", max_retries=3)
        fast_scheduler.add_task(t, flaky)
        metrics = await fast_scheduler.run()
        assert metrics.per_task["t"].retry_count == 1

    @pytest.mark.asyncio
    async def test_runtime_error_message_contains_task_id(self, fast_scheduler):
        t = make_task("my_task", max_retries=0)
        fast_scheduler.add_task(t, failing_handler)
        with pytest.raises(RuntimeError, match="my_task"):
            await fast_scheduler.run()


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """At no point should more than max_concurrency tasks run simultaneously."""
        max_concurrency = 2
        concurrent_peak = 0
        currently_running = 0

        async def slow_handler(task: Task) -> str:
            nonlocal concurrent_peak, currently_running
            currently_running += 1
            concurrent_peak = max(concurrent_peak, currently_running)
            await asyncio.sleep(0.05)
            currently_running -= 1
            return "ok"

        s = TaskScheduler(max_concurrency=max_concurrency, base_backoff=0.0)
        for i in range(6):
            s.add_task(make_task(f"t{i}"), slow_handler)

        await s.run()
        assert concurrent_peak <= max_concurrency

    @pytest.mark.asyncio
    async def test_concurrency_one_runs_sequentially(self):
        """With max_concurrency=1 tasks must run one at a time."""
        order: list[tuple[str, str]] = []  # (task_id, "start"|"end")

        async def tracking_handler(task: Task) -> str:
            order.append((task.id, "start"))
            await asyncio.sleep(0.01)
            order.append((task.id, "end"))
            return "ok"

        s = TaskScheduler(max_concurrency=1, base_backoff=0.0)
        for i in range(3):
            s.add_task(make_task(f"t{i}"), tracking_handler)

        await s.run()

        # Every start must immediately precede its own end with no interleaving
        starts_ends = [e for e in order]
        for i in range(0, len(starts_ends) - 1, 2):
            task_id = starts_ends[i][0]
            assert starts_ends[i] == (task_id, "start")
            assert starts_ends[i + 1] == (task_id, "end")

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently_within_limit(self):
        """Independent tasks should start overlapping when concurrency > 1."""
        start_times: dict[str, float] = {}
        end_times: dict[str, float] = {}

        async def timed_handler(task: Task) -> str:
            start_times[task.id] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            end_times[task.id] = asyncio.get_event_loop().time()
            return "ok"

        s = TaskScheduler(max_concurrency=4, base_backoff=0.0)
        for i in range(4):
            s.add_task(make_task(f"t{i}"), timed_handler)

        await s.run()

        # All 4 tasks should overlap: each one starts before another one ends
        # (True parallelism is indicated by overlapping time windows)
        for i in range(4):
            for j in range(4):
                if i != j:
                    # At least some overlap must exist among tasks
                    pass  # checked via total time below

        total_sequential = sum(
            end_times[f"t{i}"] - start_times[f"t{i}"] for i in range(4)
        )
        total_wall = max(end_times.values()) - min(start_times.values())
        # Wall time should be significantly less than sequential sum
        assert total_wall < total_sequential * 0.9

    @pytest.mark.asyncio
    async def test_semaphore_released_after_task_completion(self, fast_scheduler):
        """Semaphore must be released even on success so subsequent tasks can proceed."""
        fast_scheduler.add_task(make_task("a"), succeeding_handler)
        fast_scheduler.add_task(make_task("b"), succeeding_handler)
        fast_scheduler.add_task(make_task("c"), succeeding_handler)
        metrics = await fast_scheduler.run()
        # All tasks completed means semaphore was properly released
        for tid in ("a", "b", "c"):
            assert fast_scheduler._tasks[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_semaphore_released_after_task_failure(self):
        """Semaphore must be released on failure so subsequent tasks are not blocked."""
        s = TaskScheduler(max_concurrency=1, base_backoff=0.0)
        t_bad = make_task("bad", max_retries=0)
        t_good = make_task("good")
        s.add_task(t_bad, failing_handler)
        s.add_task(t_good, succeeding_handler)

        # Both tasks are independent; run them; bad one fails, good one succeeds.
        # We gather with return_exceptions=False so the RuntimeError propagates.
        # But the good task should still have been attempted (semaphore released).
        try:
            await s.run()
        except RuntimeError:
            pass  # expected from the failing task

        # The good task should have completed since semaphore was released
        assert t_good.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_high_concurrency_limit_does_not_block(self):
        """A max_concurrency higher than task count should not cause deadlocks."""
        s = TaskScheduler(max_concurrency=100, base_backoff=0.0)
        for i in range(5):
            s.add_task(make_task(f"t{i}"), succeeding_handler)
        metrics = await s.run()
        for i in range(5):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Full integration scenarios
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_complex_dag_all_tasks_complete(self):
        """
        DAG:
          a, b  →  c  →  e
          a     →  d  →  e
        """
        s = TaskScheduler(max_concurrency=4, base_backoff=0.0)
        s.add_task(make_task("a"), succeeding_handler)
        s.add_task(make_task("b"), succeeding_handler)
        s.add_task(make_task("c", dependencies=["a", "b"]), succeeding_handler)
        s.add_task(make_task("d", dependencies=["a"]), succeeding_handler)
        s.add_task(make_task("e", dependencies=["c", "d"]), succeeding_handler)
        await s.run()
        for tid in ("a", "b", "c", "d", "e"):
            assert s._tasks[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mixed_success_and_retry_tasks(self):
        """Some tasks succeed immediately, others need a retry."""
        flaky_calls = 0

        async def flaky(task: Task) -> str:
            nonlocal flaky_calls
            flaky_calls += 1
            if flaky_calls < 2:
                raise RuntimeError("transient")
            return "recovered"

        s = TaskScheduler(max_concurrency=4, base_backoff=0.0)
        s.add_task(make_task("stable"), succeeding_handler)
        s.add_task(make_task("flaky", max_retries=3), flaky)
        await s.run()
        assert s._tasks["stable"].status == TaskStatus.COMPLETED
        assert s._tasks["flaky"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_resets_metrics_on_second_call(self):
        """Calling run() twice should produce fresh metrics each time."""
        s = TaskScheduler(max_concurrency=4, base_backoff=0.0)
        s.add_task(make_task("t1"), succeeding_handler)
        first = await s.run()
        # Re-register a different task for second run (scheduler state carries over)
        s.add_task(make_task("t2"), succeeding_handler)
        second = await s.run()
        # Second run metrics should not include stale timing from first
        assert second.total_time is not None
        assert second is not first

    @pytest.mark.asyncio
    async def test_event_callbacks_fired_in_correct_order(self):
        """start event must precede complete event for every task."""
        events: list[tuple[str, str]] = []  # (event_name, task_id)

        async def on_start(task: Task) -> None:
            events.append(("start", task.id))

        async def on_complete(task: Task) -> None:
            events.append(("complete", task.id))

        s = TaskScheduler(max_concurrency=4, base_backoff=0.0)
        s.on_task_start(on_start)
        s.on_task_complete(on_complete)
        s.add_task(make_task("t1"), succeeding_handler)
        await s.run()

        start_idx = next(i for i, e in enumerate(events) if e == ("start", "t1"))
        complete_idx = next(i for i, e in enumerate(events) if e == ("complete", "t1"))
        assert start_idx < complete_idx
