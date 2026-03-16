"""
test_task_scheduler.py - Comprehensive pytest test suite for task_scheduler.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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
    task_id: str,
    name: str = None,
    priority: int = 5,
    dependencies: list = None,
    max_retries: int = 3,
    fn=None,
) -> Task:
    """Factory for Task instances with sensible defaults."""
    if name is None:
        name = task_id
    if dependencies is None:
        dependencies = []
    if fn is None:
        fn = AsyncMock(return_value=f"result_{task_id}")
    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies,
        max_retries=max_retries,
        _fn=fn,
    )


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_task_creation(self):
        t = Task(id="t1", name="Task 1", priority=5)
        assert t.id == "t1"
        assert t.name == "Task 1"
        assert t.priority == 5
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.dependencies == []
        assert t.result is None

    def test_priority_lower_bound(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=0)

    def test_priority_upper_bound(self):
        with pytest.raises(ValueError):
            Task(id="t", name="t", priority=11)

    def test_priority_boundaries_valid(self):
        t_low = Task(id="t1", name="t1", priority=1)
        t_high = Task(id="t2", name="t2", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10

    def test_default_dependencies_are_independent(self):
        t1 = Task(id="t1", name="t1", priority=5)
        t2 = Task(id="t2", name="t2", priority=5)
        t1.dependencies.append("x")
        assert t2.dependencies == []


# ---------------------------------------------------------------------------
# Basic task registration
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_single_task(self):
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task)
        assert "t1" in scheduler._tasks

    def test_add_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"))
        with pytest.raises(ValueError, match="t1"):
            scheduler.add_task(make_task("t1"))

    def test_add_multiple_tasks(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(make_task(f"t{i}"))
        assert len(scheduler._tasks) == 5


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(return_value=42)
        task = make_task("t1", fn=fn)
        scheduler.add_task(task)

        await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == 42
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_complete(self):
        scheduler = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(4)]
        for t in tasks:
            scheduler.add_task(t)

        await scheduler.run()

        for t in tasks:
            assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(return_value="hello")
        task = make_task("t1", fn=fn)
        scheduler.add_task(task)

        await scheduler.run()

        assert task.result == "hello"

    @pytest.mark.asyncio
    async def test_task_without_fn_fails(self):
        scheduler = TaskScheduler()
        task = Task(id="t1", name="t1", priority=5, _fn=None)
        scheduler.add_task(task)

        await scheduler.run()

        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_returns_scheduler_metrics(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"))

        metrics = await scheduler.run()

        assert isinstance(metrics, SchedulerMetrics)
        assert metrics.total_elapsed >= 0


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_get_execution_plan_no_deps(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b"))
        plan = scheduler.get_execution_plan()
        # Both independent tasks in first group
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_get_execution_plan_linear_chain(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["b"]))
        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_get_execution_plan_diamond(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a"))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["a"]))
        scheduler.add_task(make_task("d", dependencies=["b", "c"]))
        plan = scheduler.get_execution_plan()
        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    @pytest.mark.asyncio
    async def test_dependency_runs_before_dependent(self):
        scheduler = TaskScheduler()
        order = []

        async def fn_a():
            order.append("a")
            return "a"

        async def fn_b():
            order.append("b")
            return "b"

        task_a = make_task("a", fn=fn_a)
        task_b = make_task("b", dependencies=["a"], fn=fn_b)
        scheduler.add_task(task_a)
        scheduler.add_task(task_b)

        await scheduler.run()

        assert order.index("a") < order.index("b")
        assert task_a.status == TaskStatus.COMPLETED
        assert task_b.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dependent_skipped_when_dependency_fails(self):
        scheduler = TaskScheduler(max_concurrency=4)
        # task_a will fail immediately (max_retries=0)
        fail_fn = AsyncMock(side_effect=RuntimeError("boom"))
        task_a = make_task("a", fn=fail_fn, max_retries=0)
        task_b = make_task("b", dependencies=["a"])
        scheduler.add_task(task_a)
        scheduler.add_task(task_b)

        await scheduler.run()

        assert task_a.status == TaskStatus.FAILED
        # task_b should remain PENDING (skipped) since its dependency failed
        assert task_b.status == TaskStatus.PENDING

    def test_unknown_dependency_raises_task_not_found(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["nonexistent"]))
        with pytest.raises(TaskNotFoundError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises_on_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["ghost"]))
        with pytest.raises(TaskNotFoundError):
            await scheduler.run()


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_self_loop(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        scheduler.add_task(make_task("c", dependencies=["b"]))
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_cycle_in_larger_graph(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("root"))
        scheduler.add_task(make_task("x", dependencies=["root"]))
        scheduler.add_task(make_task("y", dependencies=["x"]))
        scheduler.add_task(make_task("z", dependencies=["y", "x"]))
        # introduce cycle between y and z
        scheduler._tasks["x"].dependencies = ["z"]
        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_cycle_raises_on_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_circular_dependency_error_contains_cycle_nodes(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]))
        scheduler.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError) as exc_info:
            scheduler.get_execution_plan()
        msg = str(exc_info.value)
        assert "a" in msg or "b" in msg


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_and_eventually_succeeds(self):
        scheduler = TaskScheduler()
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "ok"

        task = make_task("t1", fn=flaky, max_retries=3)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            scheduler.add_task(task)
            await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "ok"
        assert task.retry_count == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries_exhausted(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(side_effect=RuntimeError("always fails"))
        task = make_task("t1", fn=fn, max_retries=2)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            scheduler.add_task(task)
            await scheduler.run()

        assert task.status == TaskStatus.FAILED
        # initial attempt + 2 retries = 3 calls total
        assert fn.await_count == 3
        assert task.retry_count == 3

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(side_effect=RuntimeError("boom"))
        task = make_task("t1", fn=fn, max_retries=0)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            scheduler.add_task(task)
            await scheduler.run()

        assert task.status == TaskStatus.FAILED
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleep_called_with_correct_delays(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        task = make_task("t1", fn=fn, max_retries=3)

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler.add_task(task)
            await scheduler.run()

        # Expected backoff: 2^0=1, 2^1=2, 2^2=4 for retries 1,2,3
        assert sleep_calls == [1, 2, 4]

    @pytest.mark.asyncio
    async def test_backoff_capped_at_32_seconds(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        task = make_task("t1", fn=fn, max_retries=10)

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler.add_task(task)
            await scheduler.run()

        assert all(d <= 32 for d in sleep_calls)
        # The cap should appear for large retry numbers
        assert 32 in sleep_calls

    @pytest.mark.asyncio
    async def test_retry_count_tracked_in_metrics(self):
        scheduler = TaskScheduler()
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        task = make_task("t1", fn=fn, max_retries=2)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            scheduler.add_task(task)
            await scheduler.run()

        assert scheduler.metrics.tasks["t1"].retry_count == 3


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_concurrency_limit_one_runs_sequentially(self):
        scheduler = TaskScheduler(max_concurrency=1)
        active = []
        peak = [0]

        async def tracked_fn(tid):
            active.append(tid)
            peak[0] = max(peak[0], len(active))
            await asyncio.sleep(0)
            active.remove(tid)
            return tid

        for i in range(4):
            task = make_task(f"t{i}", fn=lambda i=i: tracked_fn(f"t{i}"))
            scheduler.add_task(task)

        await scheduler.run()

        assert peak[0] == 1

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected_under_load(self):
        limit = 3
        scheduler = TaskScheduler(max_concurrency=limit)
        active = []
        peak = [0]
        lock = asyncio.Lock()

        async def tracked_fn():
            async with lock:
                active.append(1)
                peak[0] = max(peak[0], len(active))
            await asyncio.sleep(0.01)
            async with lock:
                active.pop()
            return "done"

        for i in range(9):
            scheduler.add_task(make_task(f"t{i}", fn=tracked_fn))

        await scheduler.run()

        assert peak[0] <= limit

    @pytest.mark.asyncio
    async def test_default_max_concurrency_is_4(self):
        scheduler = TaskScheduler()
        assert scheduler._max_concurrency == 4

    @pytest.mark.asyncio
    async def test_all_tasks_complete_regardless_of_concurrency_limit(self):
        scheduler = TaskScheduler(max_concurrency=2)
        task_ids = [f"t{i}" for i in range(10)]
        for tid in task_ids:
            scheduler.add_task(make_task(tid))

        await scheduler.run()

        for tid in task_ids:
            assert scheduler._tasks[tid].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Observer pattern / lifecycle callbacks
# ---------------------------------------------------------------------------

class TestObserverCallbacks:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self):
        scheduler = TaskScheduler()
        started = []
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.add_task(make_task("t1"))

        await scheduler.run()

        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self):
        scheduler = TaskScheduler()
        completed = []
        scheduler.on_task_complete(lambda t: completed.append(t.id))
        scheduler.add_task(make_task("t1"))

        await scheduler.run()

        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called_with_exception(self):
        scheduler = TaskScheduler()
        failures = []

        def on_fail(task, exc):
            failures.append((task.id, exc))

        scheduler.on_task_fail(on_fail)
        fn = AsyncMock(side_effect=RuntimeError("oops"))
        scheduler.add_task(make_task("t1", fn=fn, max_retries=0))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await scheduler.run()

        assert len(failures) == 1
        task_id, exc = failures[0]
        assert task_id == "t1"
        assert isinstance(exc, RuntimeError)

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self):
        scheduler = TaskScheduler()
        log = []
        scheduler.on_task_start(lambda t: log.append(("start", t.id)))
        scheduler.on_task_start(lambda t: log.append(("start2", t.id)))
        scheduler.add_task(make_task("t1"))

        await scheduler.run()

        starts = [e for e in log if e[1] == "t1"]
        assert len(starts) == 2

    @pytest.mark.asyncio
    async def test_on_task_start_not_called_for_failed_dependency_skip(self):
        scheduler = TaskScheduler()
        started = []
        scheduler.on_task_start(lambda t: started.append(t.id))

        fail_fn = AsyncMock(side_effect=RuntimeError("fail"))
        scheduler.add_task(make_task("a", fn=fail_fn, max_retries=0))
        scheduler.add_task(make_task("b", dependencies=["a"]))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await scheduler.run()

        assert "b" not in started


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"))

        metrics = await scheduler.run()

        assert "t1" in metrics.tasks
        assert metrics.tasks["t1"].elapsed >= 0
        assert metrics.total_elapsed >= 0

    @pytest.mark.asyncio
    async def test_metrics_start_before_end(self):
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"))

        metrics = await scheduler.run()

        tm = metrics.tasks["t1"]
        assert tm.end_time >= tm.start_time

    @pytest.mark.asyncio
    async def test_metrics_total_elapsed_covers_all_tasks(self):
        scheduler = TaskScheduler()
        for i in range(3):
            scheduler.add_task(make_task(f"t{i}"))

        metrics = await scheduler.run()

        assert metrics.total_end >= metrics.total_start
        assert len(metrics.tasks) == 3
