"""
Comprehensive tests for task_scheduler module.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from task_scheduler import (
    CircularDependencyError,
    ExecutionMetrics,
    Task,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _success_coro(task: Task) -> str:
    """Simple coroutine that succeeds immediately."""
    return f"{task.name}-done"


async def _slow_coro(task: Task) -> str:
    """Coroutine that takes a short while to complete."""
    await asyncio.sleep(0.05)
    return f"{task.name}-slow-done"


def _make_failing_coro(fail_times: int):
    """Return a coroutine factory that fails *fail_times* then succeeds."""
    call_count = 0

    async def coro(task: Task) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise RuntimeError(f"Intentional failure #{call_count}")
        return f"{task.name}-recovered"

    return coro


def _make_always_failing_coro():
    """Return a coroutine that always raises."""
    async def coro(task: Task) -> str:
        raise RuntimeError("Always fails")
    return coro


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_default_values(self):
        t = Task(id="1", name="test")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_validation_low(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="1", name="bad", priority=0)

    def test_priority_validation_high(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="1", name="bad", priority=11)

    def test_valid_boundary_priorities(self):
        t1 = Task(id="1", name="low", priority=1)
        t10 = Task(id="2", name="high", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10


# ---------------------------------------------------------------------------
# ExecutionMetrics tests
# ---------------------------------------------------------------------------

class TestExecutionMetrics:
    def test_defaults(self):
        m = ExecutionMetrics()
        assert m.total_time == 0.0
        assert m.per_task_time == {}
        assert m.retry_counts == {}


# ---------------------------------------------------------------------------
# TaskScheduler — registration
# ---------------------------------------------------------------------------

class TestSchedulerRegistration:
    def test_add_task(self):
        s = TaskScheduler()
        t = Task(id="a", name="alpha")
        s.add_task(t, _success_coro)
        assert s.get_task("a") is t

    def test_duplicate_task_raises(self):
        s = TaskScheduler()
        t = Task(id="a", name="alpha")
        s.add_task(t, _success_coro)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="alpha2"), _success_coro)

    def test_get_task_missing(self):
        s = TaskScheduler()
        assert s.get_task("nope") is None

    def test_get_all_tasks_empty(self):
        s = TaskScheduler()
        assert s.get_all_tasks() == []

    def test_get_all_tasks(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="alpha"), _success_coro)
        s.add_task(Task(id="b", name="beta"), _success_coro)
        assert len(s.get_all_tasks()) == 2


# ---------------------------------------------------------------------------
# TaskScheduler — observer / events
# ---------------------------------------------------------------------------

class TestSchedulerEvents:
    def test_on_task_start_event(self):
        s = TaskScheduler()
        started: list[str] = []
        s.on("on_task_start", lambda t: started.append(t.id))
        s.add_task(Task(id="a", name="alpha"), _success_coro)
        asyncio.run(s.execute_all())
        assert "a" in started

    def test_on_task_complete_event(self):
        s = TaskScheduler()
        completed: list[str] = []
        s.on("on_task_complete", lambda t: completed.append(t.id))
        s.add_task(Task(id="a", name="alpha"), _success_coro)
        asyncio.run(s.execute_all())
        assert "a" in completed

    def test_on_task_fail_event(self):
        s = TaskScheduler()
        failed: list[str] = []
        s.on("on_task_fail", lambda t: failed.append(t.id))
        s.add_task(
            Task(id="a", name="alpha", max_retries=0),
            _make_always_failing_coro(),
        )
        asyncio.run(s.execute_all())
        assert "a" in failed

    def test_multiple_listeners(self):
        s = TaskScheduler()
        log1: list[str] = []
        log2: list[str] = []
        s.on("on_task_complete", lambda t: log1.append(t.id))
        s.on("on_task_complete", lambda t: log2.append(t.id))
        s.add_task(Task(id="a", name="alpha"), _success_coro)
        asyncio.run(s.execute_all())
        assert log1 == ["a"]
        assert log2 == ["a"]


# ---------------------------------------------------------------------------
# TaskScheduler — dependency graph
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    def test_unknown_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="alpha", dependencies=["missing"]),
            _success_coro,
        )
        with pytest.raises(ValueError, match="unknown task"):
            asyncio.run(s.execute_all())

    def test_circular_dependency_raises(self):
        s = TaskScheduler()
        s.add_task(
            Task(id="a", name="alpha", dependencies=["b"]),
            _success_coro,
        )
        s.add_task(
            Task(id="b", name="beta", dependencies=["a"]),
            _success_coro,
        )
        with pytest.raises(CircularDependencyError):
            asyncio.run(s.execute_all())

    def test_three_node_cycle(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), _success_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _success_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _success_coro)
        with pytest.raises(CircularDependencyError):
            asyncio.run(s.execute_all())

    def test_linear_dependency_chain(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _success_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _success_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), _success_coro)
        asyncio.run(s.execute_all())
        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("b").status == TaskStatus.COMPLETED
        assert s.get_task("c").status == TaskStatus.COMPLETED

    def test_diamond_dependency(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _success_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), _success_coro)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), _success_coro)
        s.add_task(
            Task(id="d", name="D", dependencies=["b", "c"]),
            _success_coro,
        )
        asyncio.run(s.execute_all())
        for tid in "abcd":
            assert s.get_task(tid).status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# TaskScheduler — execution basics
# ---------------------------------------------------------------------------

class TestExecution:
    def test_empty_scheduler(self):
        s = TaskScheduler()
        metrics = asyncio.run(s.execute_all())
        assert isinstance(metrics, ExecutionMetrics)

    def test_single_task_completes(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="alpha"), _success_coro)
        asyncio.run(s.execute_all())
        t = s.get_task("a")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "alpha-done"

    def test_result_stored(self):
        s = TaskScheduler()
        s.add_task(Task(id="x", name="echo"), _success_coro)
        asyncio.run(s.execute_all())
        assert s.get_task("x").result == "echo-done"

    def test_metrics_populated(self):
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), _success_coro)
        s.add_task(Task(id="b", name="B"), _success_coro)
        metrics = asyncio.run(s.execute_all())
        assert metrics.total_time > 0
        assert "a" in metrics.per_task_time
        assert "b" in metrics.per_task_time
        assert metrics.retry_counts["a"] == 0
        assert metrics.retry_counts["b"] == 0


# ---------------------------------------------------------------------------
# TaskScheduler — retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_retry_then_succeed(self):
        s = TaskScheduler(base_backoff=0.01)
        coro = _make_failing_coro(fail_times=2)
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        asyncio.run(s.execute_all())
        t = s.get_task("a")
        assert t.status == TaskStatus.COMPLETED
        assert t.retry_count == 2
        assert t.result == "A-recovered"

    def test_exceed_max_retries(self):
        s = TaskScheduler(base_backoff=0.01)
        s.add_task(
            Task(id="a", name="A", max_retries=2),
            _make_always_failing_coro(),
        )
        asyncio.run(s.execute_all())
        t = s.get_task("a")
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # tried original + 2 retries => count=3

    def test_zero_retries_fails_immediately(self):
        s = TaskScheduler(base_backoff=0.01)
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        asyncio.run(s.execute_all())
        t = s.get_task("a")
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 1

    def test_retry_count_in_metrics(self):
        s = TaskScheduler(base_backoff=0.01)
        coro = _make_failing_coro(fail_times=1)
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        metrics = asyncio.run(s.execute_all())
        assert metrics.retry_counts["a"] == 1


# ---------------------------------------------------------------------------
# TaskScheduler — concurrency control
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrency_limit_respected(self):
        """At most max_concurrency tasks should run simultaneously."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracking_coro(task: Task) -> str:
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return "ok"

        s = TaskScheduler(max_concurrency=2)
        for i in range(6):
            s.add_task(Task(id=str(i), name=f"T{i}"), tracking_coro)

        asyncio.run(s.execute_all())
        assert max_concurrent <= 2

    def test_all_tasks_complete_with_low_concurrency(self):
        s = TaskScheduler(max_concurrency=1)
        for i in range(4):
            s.add_task(Task(id=str(i), name=f"T{i}"), _success_coro)
        asyncio.run(s.execute_all())
        for i in range(4):
            assert s.get_task(str(i)).status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# TaskScheduler — priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_higher_priority_starts_first(self):
        """With max_concurrency=1, tasks should execute highest-priority first."""
        order: list[str] = []

        async def recording_coro(task: Task) -> str:
            order.append(task.id)
            return "ok"

        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="lo", name="Low", priority=1), recording_coro)
        s.add_task(Task(id="hi", name="High", priority=10), recording_coro)
        s.add_task(Task(id="mid", name="Mid", priority=5), recording_coro)
        asyncio.run(s.execute_all())

        assert order == ["hi", "mid", "lo"]


# ---------------------------------------------------------------------------
# TaskScheduler — dependent failure propagation
# ---------------------------------------------------------------------------

class TestFailurePropagation:
    def test_dependent_fails_when_dependency_fails(self):
        s = TaskScheduler(base_backoff=0.01)
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _success_coro,
        )
        asyncio.run(s.execute_all())
        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED

    def test_transitive_failure_propagation(self):
        s = TaskScheduler(base_backoff=0.01)
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _success_coro,
        )
        s.add_task(
            Task(id="c", name="C", dependencies=["b"]),
            _success_coro,
        )
        asyncio.run(s.execute_all())
        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED
        assert s.get_task("c").status == TaskStatus.FAILED

    def test_independent_tasks_unaffected_by_failure(self):
        s = TaskScheduler(base_backoff=0.01)
        s.add_task(
            Task(id="a", name="A", max_retries=0),
            _make_always_failing_coro(),
        )
        s.add_task(Task(id="b", name="B"), _success_coro)
        asyncio.run(s.execute_all())
        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.COMPLETED
