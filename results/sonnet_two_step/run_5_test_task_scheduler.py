"""
test_task_scheduler.py

Comprehensive pytest test suite for task_scheduler.py covering:
- Basic task execution
- Dependency resolution
- Circular dependency detection
- Retry logic with exponential backoff
- Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

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

def make_task(
    task_id: str,
    name: str = None,
    priority: int = 5,
    dependencies: List[str] = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


async def success_fn(task: Task):
    return f"result:{task.id}"


async def fail_fn(task: Task):
    raise RuntimeError(f"task {task.id} failed")


def make_scheduler(max_concurrency: int = 4, base_retry_delay: float = 0.0) -> TaskScheduler:
    return TaskScheduler(max_concurrency=max_concurrency, base_retry_delay=base_retry_delay)


# ---------------------------------------------------------------------------
# Task dataclass validation
# ---------------------------------------------------------------------------

class TestTaskDataclass:
    def test_valid_priority_boundaries(self):
        t1 = make_task("t", priority=1)
        t10 = make_task("t", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_low(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=0)

    def test_invalid_priority_high(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_zero(self):
        t = make_task("t")
        assert t.retry_count == 0

    def test_result_is_none_by_default(self):
        t = make_task("t")
        assert t.result is None

    def test_default_dependencies_empty(self):
        t = make_task("t")
        assert t.dependencies == []


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_when_not_started(self):
        m = TaskMetrics(task_id="t")
        assert m.elapsed is None

    def test_elapsed_none_when_not_finished(self):
        m = TaskMetrics(task_id="t", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed_correctly(self):
        m = TaskMetrics(task_id="t", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_elapsed_none_when_not_run(self):
        m = SchedulerMetrics()
        assert m.total_elapsed is None

    def test_total_elapsed_computed(self):
        m = SchedulerMetrics(total_start_time=0.0, total_end_time=5.0)
        assert m.total_elapsed == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "result:a"

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self):
        scheduler = make_scheduler()
        t = make_task("a")

        async def fn(task):
            return 42

        scheduler.register(t, fn)
        await scheduler.run()
        assert t.result == 42

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        metrics = await scheduler.run()

        assert metrics.total_start_time is not None
        assert metrics.total_end_time is not None
        assert metrics.total_elapsed is not None
        assert metrics.total_elapsed >= 0

    @pytest.mark.asyncio
    async def test_per_task_metrics_populated(self):
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        metrics = await scheduler.run()

        tm = metrics.tasks["a"]
        assert tm.start_time is not None
        assert tm.end_time is not None
        assert tm.elapsed >= 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        scheduler = make_scheduler()
        tasks = [make_task(tid) for tid in ["a", "b", "c"]]
        for t in tasks:
            scheduler.register(t, success_fn)

        await scheduler.run()

        for t in tasks:
            assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_without_error(self):
        scheduler = make_scheduler()
        metrics = await scheduler.run()
        assert metrics.total_elapsed is not None

    @pytest.mark.asyncio
    async def test_get_task_returns_registered_task(self):
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)
        assert scheduler.get_task("a") is t

    def test_get_task_raises_for_unknown_id(self):
        scheduler = make_scheduler()
        with pytest.raises(TaskNotFoundError):
            scheduler.get_task("nonexistent")

    @pytest.mark.asyncio
    async def test_tasks_property_returns_all_registered(self):
        scheduler = make_scheduler()
        t1 = make_task("a")
        t2 = make_task("b")
        scheduler.register(t1, success_fn)
        scheduler.register(t2, success_fn)

        tasks = scheduler.tasks
        assert set(tasks.keys()) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_tasks_property_is_copy(self):
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        snapshot = scheduler.tasks
        snapshot["x"] = make_task("x")
        # internal dict should be unaffected
        assert "x" not in scheduler.tasks


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_runs_after_dependency(self):
        order = []
        scheduler = make_scheduler()

        async def record(task):
            order.append(task.id)

        t_a = make_task("a")
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, record)
        scheduler.register(t_b, record)

        await scheduler.run()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_chain_a_b_c_executes_in_order(self):
        order = []
        scheduler = make_scheduler()

        async def record(task):
            order.append(task.id)

        t_a = make_task("a")
        t_b = make_task("b", dependencies=["a"])
        t_c = make_task("c", dependencies=["b"])

        for t in [t_a, t_b, t_c]:
            scheduler.register(t, record)

        await scheduler.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_multiple_dependencies_all_must_complete(self):
        order = []
        scheduler = make_scheduler()

        async def record(task):
            order.append(task.id)

        t_a = make_task("a")
        t_b = make_task("b")
        t_c = make_task("c", dependencies=["a", "b"])

        for t in [t_a, t_b, t_c]:
            scheduler.register(t, record)

        await scheduler.run()
        assert order.index("c") > order.index("a")
        assert order.index("c") > order.index("b")

    def test_unknown_dependency_raises_task_not_found(self):
        scheduler = make_scheduler()
        t = make_task("b", dependencies=["nonexistent"])
        scheduler.register(t, success_fn)

        with pytest.raises(TaskNotFoundError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_execution_plan_groups_independent_tasks(self):
        scheduler = make_scheduler()
        t_a = make_task("a")
        t_b = make_task("b")
        t_c = make_task("c", dependencies=["a", "b"])

        for t in [t_a, t_b, t_c]:
            scheduler.register(t, success_fn)

        plan = scheduler.get_execution_plan()
        # a and b should be in the same (first) level
        first_level_ids = {t.id for t in plan[0]}
        assert first_level_ids == {"a", "b"}
        second_level_ids = {t.id for t in plan[1]}
        assert second_level_ids == {"c"}

    @pytest.mark.asyncio
    async def test_priority_order_within_level(self):
        scheduler = make_scheduler()
        t_low = make_task("low", priority=1)
        t_high = make_task("high", priority=9)

        scheduler.register(t_low, success_fn)
        scheduler.register(t_high, success_fn)

        plan = scheduler.get_execution_plan()
        # Both tasks are in the first (only) level; high priority comes first
        assert plan[0][0].id == "high"

    @pytest.mark.asyncio
    async def test_failed_dependency_causes_dependent_to_be_skipped(self):
        scheduler = make_scheduler(base_retry_delay=0.0)

        t_a = make_task("a", max_retries=1)
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, fail_fn)
        scheduler.register(t_b, success_fn)

        await scheduler.run()

        assert t_a.status == TaskStatus.FAILED
        assert t_b.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_unrelated_tasks_not_affected_by_failure(self):
        scheduler = make_scheduler(base_retry_delay=0.0)

        t_a = make_task("a", max_retries=1)
        t_b = make_task("b")  # no dependency on a

        scheduler.register(t_a, fail_fn)
        scheduler.register(t_b, success_fn)

        await scheduler.run()

        assert t_a.status == TaskStatus.FAILED
        assert t_b.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_self_referencing_task(self):
        scheduler = make_scheduler()
        t = make_task("a", dependencies=["a"])
        scheduler.register(t, success_fn)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_task_cycle(self):
        scheduler = make_scheduler()
        t_a = make_task("a", dependencies=["b"])
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, success_fn)
        scheduler.register(t_b, success_fn)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_task_cycle(self):
        scheduler = make_scheduler()
        t_a = make_task("a", dependencies=["c"])
        t_b = make_task("b", dependencies=["a"])
        t_c = make_task("c", dependencies=["b"])

        for t in [t_a, t_b, t_c]:
            scheduler.register(t, success_fn)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_cycle_in_larger_graph_with_acyclic_nodes(self):
        scheduler = make_scheduler()
        # independent chain: x -> y
        t_x = make_task("x")
        t_y = make_task("y", dependencies=["x"])
        # cycle: a -> b -> a
        t_a = make_task("a", dependencies=["b"])
        t_b = make_task("b", dependencies=["a"])

        for t in [t_x, t_y, t_a, t_b]:
            scheduler.register(t, success_fn)

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        scheduler = make_scheduler()
        t_a = make_task("a", dependencies=["b"])
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, success_fn)
        scheduler.register(t_b, success_fn)

        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_no_false_positive_on_diamond_dependency(self):
        """a -> b, a -> c, b -> d, c -> d  (diamond, not a cycle)"""
        scheduler = make_scheduler()
        t_a = make_task("a")
        t_b = make_task("b", dependencies=["a"])
        t_c = make_task("c", dependencies=["a"])
        t_d = make_task("d", dependencies=["b", "c"])

        for t in [t_a, t_b, t_c, t_d]:
            scheduler.register(t, success_fn)

        plan = scheduler.get_execution_plan()
        all_ids = {t.id for level in plan for t in level}
        assert all_ids == {"a", "b", "c", "d"}

    def test_circular_dependency_error_message_contains_cycle_ids(self):
        scheduler = make_scheduler()
        t_a = make_task("a", dependencies=["b"])
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, success_fn)
        scheduler.register(t_b, success_fn)

        with pytest.raises(CircularDependencyError, match="a|b"):
            scheduler.get_execution_plan()


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retried_up_to_max_retries(self):
        call_count = 0
        scheduler = make_scheduler(base_retry_delay=0.0)

        async def flaky(task):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        t = make_task("a", max_retries=3)
        scheduler.register(t, flaky)
        await scheduler.run()

        assert call_count == 3
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3

    @pytest.mark.asyncio
    async def test_task_succeeds_on_retry(self):
        attempts = []
        scheduler = make_scheduler(base_retry_delay=0.0)

        async def eventually_ok(task):
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("not yet")
            return "ok"

        t = make_task("a", max_retries=3)
        scheduler.register(t, eventually_ok)
        await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "ok"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_retry_count_increments_correctly(self):
        scheduler = make_scheduler(base_retry_delay=0.0)
        t = make_task("a", max_retries=3)
        scheduler.register(t, fail_fn)
        await scheduler.run()
        assert t.retry_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self):
        scheduler = make_scheduler(base_retry_delay=0.0)
        t = make_task("a", max_retries=2)
        scheduler.register(t, fail_fn)
        metrics = await scheduler.run()
        assert metrics.tasks["a"].retry_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay_increases(self):
        """Verify that asyncio.sleep is called with increasing delays."""
        sleep_calls = []
        scheduler = make_scheduler(base_retry_delay=1.0)

        original_sleep = asyncio.sleep

        async def capture_sleep(delay):
            sleep_calls.append(delay)

        async def always_fail(task):
            raise RuntimeError("fail")

        t = make_task("a", max_retries=4)
        scheduler.register(t, always_fail)

        with patch("task_scheduler.asyncio.sleep", side_effect=capture_sleep):
            await scheduler.run()

        # Delays should be 1, 2, 4 (base * 2^0, 2^1, 2^2) for 3 sleeps
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)
        assert sleep_calls[2] == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_max_retries_one_means_single_attempt(self):
        """max_retries=1 → called once, immediately fails."""
        call_count = 0
        scheduler = make_scheduler(base_retry_delay=0.0)

        async def counting_fail(task):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        t = make_task("a", max_retries=1)
        scheduler.register(t, counting_fail)
        await scheduler.run()

        assert call_count == 1
        assert t.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_no_sleep_on_final_retry(self):
        """asyncio.sleep should not be called after the last retry."""
        sleep_calls = []

        async def counting_fail(task):
            raise RuntimeError("fail")

        scheduler = make_scheduler(base_retry_delay=1.0)
        t = make_task("a", max_retries=2)
        scheduler.register(t, counting_fail)

        with patch("task_scheduler.asyncio.sleep", side_effect=lambda d: sleep_calls.append(d)):
            await scheduler.run()

        # max_retries=2 means 2 total attempts, 1 sleep between them
        assert len(sleep_calls) == 1


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_concurrency_limit_of_one_serialises_tasks(self):
        """With max_concurrency=1, tasks in the same level run one at a time."""
        active = []
        peak = []

        async def track(task):
            active.append(task.id)
            peak.append(len(active))
            await asyncio.sleep(0)
            active.remove(task.id)

        scheduler = make_scheduler(max_concurrency=1)
        for tid in ["a", "b", "c", "d"]:
            scheduler.register(make_task(tid), track)

        await scheduler.run()
        assert max(peak) == 1

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At most max_concurrency tasks should be running simultaneously."""
        max_concurrency = 2
        active = []
        peak = [0]

        async def track(task):
            active.append(task.id)
            peak[0] = max(peak[0], len(active))
            await asyncio.sleep(0.01)
            active.remove(task.id)

        scheduler = make_scheduler(max_concurrency=max_concurrency)
        for tid in ["a", "b", "c", "d", "e"]:
            scheduler.register(make_task(tid), track)

        await scheduler.run()
        assert peak[0] <= max_concurrency

    @pytest.mark.asyncio
    async def test_all_tasks_complete_regardless_of_concurrency_limit(self):
        scheduler = make_scheduler(max_concurrency=2)
        task_ids = [str(i) for i in range(8)]
        for tid in task_ids:
            scheduler.register(make_task(tid), success_fn)

        await scheduler.run()

        for tid in task_ids:
            assert scheduler.get_task(tid).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_high_concurrency_limit_allows_parallel_execution(self):
        """All independent tasks can run truly in parallel when limit is high."""
        started: List[str] = []
        barrier = asyncio.Event()

        async def wait_for_barrier(task):
            started.append(task.id)
            await barrier.wait()
            return task.id

        num_tasks = 5
        scheduler = make_scheduler(max_concurrency=num_tasks)
        for i in range(num_tasks):
            scheduler.register(make_task(str(i)), wait_for_barrier)

        run_coro = asyncio.ensure_future(scheduler.run())

        # Give tasks time to start
        await asyncio.sleep(0.05)
        assert len(started) == num_tasks, (
            f"Expected all {num_tasks} tasks to start concurrently, "
            f"but only {len(started)} started"
        )
        barrier.set()
        await run_coro

    @pytest.mark.asyncio
    async def test_dependency_levels_execute_serially(self):
        """Even with high concurrency, level N+1 must wait for level N."""
        level_a_done = False

        async def level_a_fn(task):
            nonlocal level_a_done
            await asyncio.sleep(0.01)
            level_a_done = True

        async def level_b_fn(task):
            assert level_a_done, "Level B started before level A finished"

        scheduler = make_scheduler(max_concurrency=10)
        t_a = make_task("a")
        t_b = make_task("b", dependencies=["a"])

        scheduler.register(t_a, level_a_fn)
        scheduler.register(t_b, level_b_fn)

        await scheduler.run()
        assert t_b.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Observer / event system
# ---------------------------------------------------------------------------

class TestEventSystem:
    @pytest.mark.asyncio
    async def test_on_task_start_fires(self):
        started = []
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        async def handler(task):
            started.append(task.id)

        scheduler.on_task_start(handler)
        await scheduler.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self):
        completed = []
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        async def handler(task):
            completed.append(task.id)

        scheduler.on_task_complete(handler)
        await scheduler.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self):
        failed = []
        scheduler = make_scheduler(base_retry_delay=0.0)
        t = make_task("a", max_retries=1)
        scheduler.register(t, fail_fn)

        async def handler(task):
            failed.append(task.id)

        scheduler.on_task_fail(handler)
        await scheduler.run()
        assert "a" in failed

    @pytest.mark.asyncio
    async def test_start_event_receives_correct_task(self):
        received = []
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        async def handler(task):
            received.append(task)

        scheduler.on_task_start(handler)
        await scheduler.run()
        assert received[0] is t

    @pytest.mark.asyncio
    async def test_handler_deregistration_with_off(self):
        from task_scheduler import EventEmitter

        emitter = EventEmitter()
        calls = []

        async def h(task):
            calls.append(task.id)

        emitter.on("evt", h)
        emitter.off("evt", h)

        dummy = make_task("x")
        await emitter.emit("evt", dummy)
        assert calls == []

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_fire(self):
        calls = []
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        async def h1(task):
            calls.append("h1")

        async def h2(task):
            calls.append("h2")

        scheduler.on_task_complete(h1)
        scheduler.on_task_complete(h2)
        await scheduler.run()

        assert "h1" in calls
        assert "h2" in calls

    @pytest.mark.asyncio
    async def test_start_fires_before_complete(self):
        order = []
        scheduler = make_scheduler()
        t = make_task("a")
        scheduler.register(t, success_fn)

        async def on_start(task):
            order.append("start")

        async def on_complete(task):
            order.append("complete")

        scheduler.on_task_start(on_start)
        scheduler.on_task_complete(on_complete)
        await scheduler.run()

        assert order == ["start", "complete"]
