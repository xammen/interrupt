"""Tests for the async task scheduler module."""

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

async def _noop() -> str:
    """A trivial coroutine that succeeds immediately."""
    return "ok"


async def _slow(seconds: float = 0.05) -> str:
    await asyncio.sleep(seconds)
    return "done"


def _make_failing_coro(fail_times: int):
    """Return a coroutine factory that fails *fail_times* then succeeds."""
    calls: dict[str, int] = {"n": 0}

    async def _coro() -> str:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(f"Intentional failure #{calls['n']}")
        return "recovered"

    return _coro


async def _always_fail() -> None:
    raise RuntimeError("permanent failure")


# ===========================================================================
# Task dataclass tests
# ===========================================================================

class TestTask:
    def test_default_values(self) -> None:
        t = Task(id="t1", name="Task 1")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=0)

    def test_priority_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=11)

    def test_valid_priority_boundaries(self) -> None:
        t_low = Task(id="t1", name="low", priority=1)
        t_high = Task(id="t2", name="high", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10


# ===========================================================================
# TaskMetrics tests
# ===========================================================================

class TestTaskMetrics:
    def test_duration(self) -> None:
        m = TaskMetrics(task_id="t1", start_time=10.0, end_time=12.5)
        assert m.duration == pytest.approx(2.5)

    def test_duration_zero(self) -> None:
        m = TaskMetrics(task_id="t1", start_time=0.0, end_time=0.0)
        assert m.duration == 0.0


# ===========================================================================
# Scheduler – registration
# ===========================================================================

class TestAddTask:
    def test_add_single_task(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        assert "a" in s.tasks

    def test_duplicate_task_id_raises(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="A2"), _noop)

    def test_get_task(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="Alpha"), _noop)
        t = s.get_task("a")
        assert t.name == "Alpha"

    def test_get_task_missing_raises(self) -> None:
        s = TaskScheduler()
        with pytest.raises(TaskNotFoundError):
            s.get_task("missing")


# ===========================================================================
# Scheduler – constructor validation
# ===========================================================================

class TestSchedulerInit:
    def test_max_concurrency_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)

    def test_max_concurrency_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            TaskScheduler(max_concurrency=-3)


# ===========================================================================
# Scheduler – dependency validation
# ===========================================================================

class TestDependencyValidation:
    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["ghost"]), _noop)
        with pytest.raises(TaskNotFoundError, match="ghost"):
            await s.run()


# ===========================================================================
# Scheduler – cycle detection
# ===========================================================================

class TestCycleDetection:
    @pytest.mark.asyncio
    async def test_self_dependency(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            await s.run()

    @pytest.mark.asyncio
    async def test_two_node_cycle(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        with pytest.raises(CircularDependencyError):
            await s.run()

    @pytest.mark.asyncio
    async def test_three_node_cycle(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _noop)
        with pytest.raises(CircularDependencyError):
            await s.run()


# ===========================================================================
# Scheduler – basic execution
# ===========================================================================

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        await s.run()
        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "ok"

    @pytest.mark.asyncio
    async def test_empty_scheduler(self) -> None:
        s = TaskScheduler()
        metrics = await s.run()
        assert isinstance(metrics, SchedulerMetrics)
        assert metrics.total_time == 0.0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self) -> None:
        s = TaskScheduler()
        for i in range(5):
            s.add_task(Task(id=f"t{i}", name=f"Task {i}"), _noop)
        await s.run()
        for i in range(5):
            assert s.get_task(f"t{i}").status == TaskStatus.COMPLETED


# ===========================================================================
# Scheduler – dependency ordering
# ===========================================================================

class TestDependencyOrdering:
    @pytest.mark.asyncio
    async def test_linear_chain(self) -> None:
        """a -> b -> c  (c depends on b, b depends on a)."""
        order: list[str] = []

        async def _track(label: str) -> str:
            order.append(label)
            return label

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), lambda: _track("a"))
        s.add_task(Task(id="b", name="B", dependencies=["a"]), lambda: _track("b"))
        s.add_task(Task(id="c", name="C", dependencies=["b"]), lambda: _track("c"))
        await s.run()

        assert order.index("a") < order.index("b") < order.index("c")

    @pytest.mark.asyncio
    async def test_diamond_dependency(self) -> None:
        """
             a
            / \\
           b   c
            \\ /
             d
        d depends on b and c; b and c depend on a.
        """
        order: list[str] = []

        async def _track(label: str) -> str:
            order.append(label)
            return label

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), lambda: _track("a"))
        s.add_task(Task(id="b", name="B", dependencies=["a"]), lambda: _track("b"))
        s.add_task(Task(id="c", name="C", dependencies=["a"]), lambda: _track("c"))
        s.add_task(
            Task(id="d", name="D", dependencies=["b", "c"]), lambda: _track("d")
        )
        await s.run()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")


# ===========================================================================
# Scheduler – priority ordering
# ===========================================================================

class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_higher_priority_starts_first_in_level(self) -> None:
        """With concurrency=1, tasks in the same level run sequentially
        and should respect priority (highest first)."""
        order: list[str] = []

        async def _track(label: str) -> str:
            order.append(label)
            return label

        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="lo", name="Low", priority=1), lambda: _track("lo"))
        s.add_task(Task(id="hi", name="High", priority=10), lambda: _track("hi"))
        s.add_task(Task(id="mid", name="Mid", priority=5), lambda: _track("mid"))
        await s.run()

        assert order == ["hi", "mid", "lo"]


# ===========================================================================
# Scheduler – retry logic
# ===========================================================================

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_succeeds_within_limit(self) -> None:
        s = TaskScheduler()
        coro = _make_failing_coro(fail_times=2)
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        await s.run()
        task = s.get_task("a")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "recovered"
        assert task.retry_count == 2  # failed twice, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=2), _always_fail)
        await s.run()
        task = s.get_task("a")
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_no_retries(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), _always_fail)
        await s.run()
        assert s.get_task("a").status == TaskStatus.FAILED


# ===========================================================================
# Scheduler – concurrency control
# ===========================================================================

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        """Ensure no more than max_concurrency tasks run simultaneously."""
        max_concurrent = 2
        concurrent_count = 0
        peak_concurrent = 0
        lock = asyncio.Lock()

        async def _tracked() -> str:
            nonlocal concurrent_count, peak_concurrent
            async with lock:
                concurrent_count += 1
                peak_concurrent = max(peak_concurrent, concurrent_count)
            await asyncio.sleep(0.03)
            async with lock:
                concurrent_count -= 1
            return "done"

        s = TaskScheduler(max_concurrency=max_concurrent)
        for i in range(6):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), _tracked)
        await s.run()

        assert peak_concurrent <= max_concurrent


# ===========================================================================
# Scheduler – observer events
# ===========================================================================

class TestObserverEvents:
    @pytest.mark.asyncio
    async def test_on_task_start_fires(self) -> None:
        started: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_start", lambda task: started.append(task.id))
        await s.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self) -> None:
        completed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", lambda task: completed.append(task.id))
        await s.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self) -> None:
        failed: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), _always_fail)
        s.on("on_task_fail", lambda task, exc: failed.append(task.id))
        await s.run()
        assert "a" in failed

    @pytest.mark.asyncio
    async def test_async_listener(self) -> None:
        results: list[str] = []

        async def _async_cb(task: Task) -> None:
            results.append(task.id)

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_start", _async_cb)
        await s.run()
        assert "a" in results

    @pytest.mark.asyncio
    async def test_multiple_listeners(self) -> None:
        log1: list[str] = []
        log2: list[str] = []
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        s.on("on_task_complete", lambda task: log1.append(task.id))
        s.on("on_task_complete", lambda task: log2.append(task.id))
        await s.run()
        assert log1 == ["a"]
        assert log2 == ["a"]


# ===========================================================================
# Scheduler – metrics
# ===========================================================================

class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        metrics = await s.run()
        assert "a" in metrics.task_metrics
        tm = metrics.task_metrics["a"]
        assert tm.duration >= 0
        assert tm.retries == 0

    @pytest.mark.asyncio
    async def test_total_time_positive(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), lambda: _slow(0.02))
        metrics = await s.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self) -> None:
        s = TaskScheduler()
        coro = _make_failing_coro(fail_times=1)
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        metrics = await s.run()
        assert metrics.task_metrics["a"].retries == 1

    @pytest.mark.asyncio
    async def test_metrics_property(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        await s.run()
        assert s.metrics is not None
        assert "a" in s.metrics.task_metrics


# ===========================================================================
# Scheduler – tasks property
# ===========================================================================

class TestTasksProperty:
    def test_tasks_returns_copy(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _noop)
        view = s.tasks
        view["b"] = Task(id="b", name="B")  # type: ignore[assignment]
        assert "b" not in s.tasks  # original unaffected


# ===========================================================================
# Integration / end-to-end
# ===========================================================================

class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        """End-to-end: dependencies, retries, events, metrics."""
        events: dict[str, list[str]] = {
            "started": [],
            "completed": [],
            "failed": [],
        }

        s = TaskScheduler(max_concurrency=2)
        s.on("on_task_start", lambda t: events["started"].append(t.id))
        s.on("on_task_complete", lambda t: events["completed"].append(t.id))
        s.on("on_task_fail", lambda t, e: events["failed"].append(t.id))

        # a (no deps), b depends on a, c depends on a, d depends on b & c
        s.add_task(Task(id="a", name="A"), _noop)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _noop)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _noop)
        s.add_task(
            Task(id="d", name="D", dependencies=["b", "c"]), _noop
        )
        # e always fails
        s.add_task(
            Task(id="e", name="E", max_retries=1), _always_fail
        )

        metrics = await s.run()

        # a, b, c, d should complete
        for tid in ("a", "b", "c", "d"):
            assert s.get_task(tid).status == TaskStatus.COMPLETED
            assert tid in events["completed"]
            assert tid in metrics.task_metrics

        # e should fail
        assert s.get_task("e").status == TaskStatus.FAILED
        assert "e" in events["failed"]
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_large_graph(self) -> None:
        """Scheduler handles a larger fan-out / fan-in graph."""
        n = 20
        s = TaskScheduler(max_concurrency=4)

        # root task
        s.add_task(Task(id="root", name="Root"), _noop)

        # fan-out: all depend on root
        for i in range(n):
            s.add_task(
                Task(id=f"mid-{i}", name=f"Mid {i}", dependencies=["root"]),
                _noop,
            )

        # fan-in: depends on all mid tasks
        s.add_task(
            Task(
                id="sink",
                name="Sink",
                dependencies=[f"mid-{i}" for i in range(n)],
            ),
            _noop,
        )

        await s.run()

        assert s.get_task("root").status == TaskStatus.COMPLETED
        for i in range(n):
            assert s.get_task(f"mid-{i}").status == TaskStatus.COMPLETED
        assert s.get_task("sink").status == TaskStatus.COMPLETED
