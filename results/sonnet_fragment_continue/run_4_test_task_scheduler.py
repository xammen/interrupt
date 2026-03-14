"""
test_task_scheduler.py - Comprehensive tests for task_scheduler.py.

Covers:
- Task dataclass validation
- TaskMetrics / SchedulerMetrics properties
- TaskScheduler registration / observer API
- Dependency validation (missing deps, circular deps)
- Priority-based ordering
- Concurrency limiting
- Retry logic (success after retries, exhaustion → FAILED)
- Observer callbacks (start / complete / fail)
- Failing observers do not crash the scheduler
- Introspection helpers (get_task, get_all_tasks, get_metrics)
- End-to-end happy-path run
"""

from __future__ import annotations

import asyncio
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

async def succeed(task: Task) -> str:
    """Trivial coroutine that succeeds immediately."""
    return f"ok:{task.id}"


async def fail_always(task: Task) -> None:
    """Coroutine that always raises."""
    raise RuntimeError("deliberate failure")


def make_task(
    tid: str,
    *,
    priority: int = 5,
    deps: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=tid,
        name=f"Task-{tid}",
        priority=priority,
        dependencies=deps or [],
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_task_creation(self) -> None:
        t = make_task("t1", priority=7)
        assert t.id == "t1"
        assert t.priority == 7
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.result is None

    def test_priority_boundary_low(self) -> None:
        t = make_task("t1", priority=1)
        assert t.priority == 1

    def test_priority_boundary_high(self) -> None:
        t = make_task("t1", priority=10)
        assert t.priority == 10

    def test_priority_too_low(self) -> None:
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=0)

    def test_priority_too_high(self) -> None:
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=11)

    def test_default_dependencies_are_independent(self) -> None:
        t1 = make_task("a")
        t2 = make_task("b")
        t1.dependencies.append("z")
        assert t2.dependencies == []


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_before_run(self) -> None:
        m = TaskMetrics(task_id="t1")
        assert m.elapsed is None

    def test_elapsed_only_start(self) -> None:
        m = TaskMetrics(task_id="t1", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed(self) -> None:
        m = TaskMetrics(task_id="t1", start_time=1.0, end_time=3.5)
        assert m.elapsed == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_time_none_before_run(self) -> None:
        sm = SchedulerMetrics()
        assert sm.total_time is None

    def test_total_time_computed(self) -> None:
        sm = SchedulerMetrics(total_start=0.0, total_end=5.0)
        assert sm.total_time == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TaskScheduler – registration
# ---------------------------------------------------------------------------

class TestTaskSchedulerRegistration:
    def test_add_task(self) -> None:
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, succeed)
        assert sched.get_task("t1") is t

    def test_duplicate_task_raises(self) -> None:
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, succeed)
        with pytest.raises(ValueError, match="already registered"):
            sched.add_task(make_task("t1"), succeed)

    def test_get_task_unknown_raises(self) -> None:
        sched = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            sched.get_task("ghost")

    def test_get_all_tasks_empty(self) -> None:
        sched = TaskScheduler()
        assert sched.get_all_tasks() == []

    def test_get_all_tasks_order(self) -> None:
        sched = TaskScheduler()
        for tid in ("a", "b", "c"):
            sched.add_task(make_task(tid), succeed)
        ids = [t.id for t in sched.get_all_tasks()]
        assert ids == ["a", "b", "c"]

    def test_metrics_entry_created_on_add(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), succeed)
        metrics = sched.get_metrics()
        assert "t1" in metrics.per_task


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("t1", deps=["ghost"]), succeed)
        with pytest.raises(TaskNotFoundError, match="ghost"):
            await sched.run()

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("a", deps=["b"]), succeed)
        sched.add_task(make_task("b", deps=["a"]), succeed)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    @pytest.mark.asyncio
    async def test_three_way_cycle_raises(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("x", deps=["z"]), succeed)
        sched.add_task(make_task("y", deps=["x"]), succeed)
        sched.add_task(make_task("z", deps=["y"]), succeed)
        with pytest.raises(CircularDependencyError):
            await sched.run()

    def test_valid_chain_does_not_raise(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("a"), succeed)
        sched.add_task(make_task("b", deps=["a"]), succeed)
        sched.add_task(make_task("c", deps=["b"]), succeed)
        # _validate_dependencies should pass without error
        sched._validate_dependencies()


# ---------------------------------------------------------------------------
# End-to-end: happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_single_task_completes(self) -> None:
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t, succeed)
        await sched.run()
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "ok:t1"

    @pytest.mark.asyncio
    async def test_chain_completes_in_order(self) -> None:
        order: list[str] = []

        async def record(task: Task) -> str:
            order.append(task.id)
            return task.id

        sched = TaskScheduler()
        sched.add_task(make_task("a"), record)
        sched.add_task(make_task("b", deps=["a"]), record)
        sched.add_task(make_task("c", deps=["b"]), record)
        await sched.run()
        assert order == ["a", "b", "c"]
        for tid in ("a", "b", "c"):
            assert sched.get_task(tid).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_independent_tasks_all_complete(self) -> None:
        sched = TaskScheduler(max_concurrency=4)
        for i in range(4):
            sched.add_task(make_task(f"t{i}"), succeed)
        await sched.run()
        for i in range(4):
            assert sched.get_task(f"t{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), succeed)
        metrics = await sched.run()
        assert metrics.total_time is not None
        assert metrics.total_time >= 0.0
        tm = metrics.per_task["t1"]
        assert tm.elapsed is not None
        assert tm.elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_diamond_dependency(self) -> None:
        """A → (B, C) → D"""
        sched = TaskScheduler()
        sched.add_task(make_task("A"), succeed)
        sched.add_task(make_task("B", deps=["A"]), succeed)
        sched.add_task(make_task("C", deps=["A"]), succeed)
        sched.add_task(make_task("D", deps=["B", "C"]), succeed)
        await sched.run()
        for tid in ("A", "B", "C", "D"):
            assert sched.get_task(tid).status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriority:
    @pytest.mark.asyncio
    async def test_higher_priority_starts_first(self) -> None:
        """With concurrency=1, priority=10 should run before priority=1."""
        start_order: list[str] = []

        async def record_start(task: Task) -> str:
            start_order.append(task.id)
            return task.id

        sched = TaskScheduler(max_concurrency=1)
        sched.add_task(make_task("low", priority=1), record_start)
        sched.add_task(make_task("high", priority=10), record_start)
        await sched.run()
        assert start_order[0] == "high"

    @pytest.mark.asyncio
    async def test_three_priorities(self) -> None:
        start_order: list[str] = []

        async def record(task: Task) -> str:
            start_order.append(task.id)
            return task.id

        sched = TaskScheduler(max_concurrency=1)
        sched.add_task(make_task("p3", priority=3), record)
        sched.add_task(make_task("p7", priority=7), record)
        sched.add_task(make_task("p5", priority=5), record)
        await sched.run()
        assert start_order == ["p7", "p5", "p3"]


# ---------------------------------------------------------------------------
# Concurrency limiting
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self) -> None:
        """Verify that at most max_concurrency tasks run simultaneously."""
        concurrent_count = 0
        max_seen = 0
        barrier = asyncio.Event()

        async def slow_task(task: Task) -> str:
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return task.id

        limit = 2
        sched = TaskScheduler(max_concurrency=limit)
        for i in range(6):
            sched.add_task(make_task(f"t{i}"), slow_task)
        await sched.run()
        assert max_seen <= limit


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_succeeds_after_retries(self) -> None:
        """Task fails twice then succeeds on third attempt."""
        call_count = 0

        async def flaky(task: Task) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        t = make_task("flaky", max_retries=3)
        sched = TaskScheduler()
        sched.add_task(t, flaky)
        await sched.run()
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "recovered"
        assert t.retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self) -> None:
        t = make_task("bad", max_retries=2)
        sched = TaskScheduler()
        sched.add_task(t, fail_always)
        await sched.run()
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_count_tracked_in_metrics(self) -> None:
        t = make_task("bad", max_retries=1)
        sched = TaskScheduler()
        sched.add_task(t, fail_always)
        metrics = await sched.run()
        assert metrics.per_task["bad"].retry_count == 2

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self) -> None:
        t = make_task("bad", max_retries=0)
        sched = TaskScheduler()
        sched.add_task(t, fail_always)
        await sched.run()
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1


# ---------------------------------------------------------------------------
# Observer / event callbacks
# ---------------------------------------------------------------------------

class TestObservers:
    @pytest.mark.asyncio
    async def test_on_task_start_called(self) -> None:
        started: list[str] = []
        sched = TaskScheduler()
        sched.on_task_start(lambda t: started.append(t.id))
        sched.add_task(make_task("t1"), succeed)
        await sched.run()
        assert "t1" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_called(self) -> None:
        completed: list[str] = []
        sched = TaskScheduler()
        sched.on_task_complete(lambda t: completed.append(t.id))
        sched.add_task(make_task("t1"), succeed)
        await sched.run()
        assert "t1" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_called(self) -> None:
        failed: list[str] = []
        sched = TaskScheduler()
        sched.on_task_fail(lambda t: failed.append(t.id))
        sched.add_task(make_task("t1", max_retries=0), fail_always)
        await sched.run()
        assert "t1" in failed

    @pytest.mark.asyncio
    async def test_multiple_observers_all_called(self) -> None:
        log1: list[str] = []
        log2: list[str] = []
        sched = TaskScheduler()
        sched.on_task_complete(lambda t: log1.append(t.id))
        sched.on_task_complete(lambda t: log2.append(t.id))
        sched.add_task(make_task("t1"), succeed)
        await sched.run()
        assert "t1" in log1
        assert "t1" in log2

    @pytest.mark.asyncio
    async def test_crashing_observer_does_not_stop_scheduler(self) -> None:
        """An observer that raises must not propagate the exception."""

        def bad_observer(task: Task) -> None:
            raise RuntimeError("observer crash")

        sched = TaskScheduler()
        sched.on_task_complete(bad_observer)
        t = make_task("t1")
        sched.add_task(t, succeed)
        # Should complete without raising
        await sched.run()
        assert t.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_start_callback_sees_running_status(self) -> None:
        statuses: list[TaskStatus] = []
        sched = TaskScheduler()
        sched.on_task_start(lambda t: statuses.append(t.status))
        sched.add_task(make_task("t1"), succeed)
        await sched.run()
        assert TaskStatus.RUNNING in statuses

    @pytest.mark.asyncio
    async def test_complete_callback_sees_completed_status(self) -> None:
        statuses: list[TaskStatus] = []
        sched = TaskScheduler()
        sched.on_task_complete(lambda t: statuses.append(t.status))
        sched.add_task(make_task("t1"), succeed)
        await sched.run()
        assert TaskStatus.COMPLETED in statuses


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    @pytest.mark.asyncio
    async def test_get_metrics_returns_same_object(self) -> None:
        sched = TaskScheduler()
        sched.add_task(make_task("t1"), succeed)
        metrics_before = sched.get_metrics()
        metrics_after = await sched.run()
        assert metrics_before is metrics_after

    def test_get_all_tasks_reflects_registration(self) -> None:
        sched = TaskScheduler()
        tasks = [make_task(f"t{i}") for i in range(3)]
        for t in tasks:
            sched.add_task(t, succeed)
        assert sched.get_all_tasks() == tasks
