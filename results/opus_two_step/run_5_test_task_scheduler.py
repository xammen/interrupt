"""
Comprehensive tests for task_scheduler.py.

Covers:
  - Task dataclass validation
  - Task registration and removal
  - Dependency resolution / topological sort
  - Circular dependency detection
  - Retry logic with exponential backoff
  - Concurrent execution respecting concurrency limits
  - Observer / event system
  - Failure propagation to downstream tasks
  - Metrics collection
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _noop_coro():
    """Return an async function that does nothing."""
    async def _fn():
        pass
    return _fn


def _result_coro(value):
    """Return an async function that returns *value*."""
    async def _fn():
        return value
    return _fn


def _fail_coro(exc: Exception):
    """Return an async function that always raises *exc*."""
    async def _fn():
        raise exc
    return _fn


def _fail_n_then_succeed(n: int, result=None, exc_factory=None):
    """Return an async function that fails *n* times then succeeds.

    Args:
        n: Number of times to raise before succeeding.
        result: Value to return on success.
        exc_factory: Callable returning the exception to raise (default RuntimeError).
    """
    call_count = {"value": 0}
    if exc_factory is None:
        exc_factory = lambda: RuntimeError("transient")

    async def _fn():
        call_count["value"] += 1
        if call_count["value"] <= n:
            raise exc_factory()
        return result

    _fn.call_count = call_count  # expose for assertions
    return _fn


# ================================================================== #
#  1. Task dataclass validation
# ================================================================== #

class TestTaskDataclass:
    def test_create_task_with_defaults(self):
        task = Task(id="t1", name="Task 1")
        assert task.id == "t1"
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.dependencies == []
        assert task.result is None

    def test_priority_lower_bound(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=0)

    def test_priority_upper_bound(self):
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="bad", priority=11)

    def test_priority_boundary_valid(self):
        t1 = Task(id="t1", name="low", priority=1)
        t10 = Task(id="t10", name="high", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10


# ================================================================== #
#  2. Basic task execution
# ================================================================== #

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        scheduler = TaskScheduler()
        task = Task(id="a", name="Alpha")
        scheduler.add_task(task, _result_coro(42))

        metrics = await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == 42
        assert "a" in metrics.per_task_time
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self):
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task {i}"),
                _result_coro(i),
            )

        await scheduler.run()

        for i in range(5):
            t = scheduler.tasks[f"t{i}"]
            assert t.status == TaskStatus.COMPLETED
            assert t.result == i

    @pytest.mark.asyncio
    async def test_task_result_stored(self):
        scheduler = TaskScheduler()
        task = Task(id="r", name="Result")
        scheduler.add_task(task, _result_coro({"key": "value"}))
        await scheduler.run()
        assert task.result == {"key": "value"}


# ================================================================== #
#  3. Task registration and removal
# ================================================================== #

class TestRegistration:
    def test_add_duplicate_task_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="x", name="X"), _noop_coro())
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(Task(id="x", name="X2"), _noop_coro())

    def test_remove_task(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="x", name="X"), _noop_coro())
        scheduler.remove_task("x")
        assert "x" not in scheduler.tasks

    def test_remove_nonexistent_task_raises(self):
        scheduler = TaskScheduler()
        with pytest.raises(KeyError, match="not found"):
            scheduler.remove_task("ghost")

    def test_tasks_property_returns_copy(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        copy = scheduler.tasks
        copy["b"] = Task(id="b", name="B")
        assert "b" not in scheduler.tasks  # original unmodified


# ================================================================== #
#  4. Dependency resolution
# ================================================================== #

class TestDependencyResolution:
    def test_linear_chain(self):
        """A -> B -> C should produce three groups of one task each."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro())
        scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _noop_coro())

        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert [t.id for t in plan[0]] == ["a"]
        assert [t.id for t in plan[1]] == ["b"]
        assert [t.id for t in plan[2]] == ["c"]

    def test_diamond_dependency(self):
        """
        A -> B
        A -> C
        B, C -> D
        Should yield groups: [A], [B, C], [D]
        """
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", priority=5), _noop_coro())
        scheduler.add_task(Task(id="b", name="B", priority=5, dependencies=["a"]), _noop_coro())
        scheduler.add_task(Task(id="c", name="C", priority=5, dependencies=["a"]), _noop_coro())
        scheduler.add_task(Task(id="d", name="D", priority=5, dependencies=["b", "c"]), _noop_coro())

        plan = scheduler.get_execution_plan()
        assert len(plan) == 3
        assert [t.id for t in plan[0]] == ["a"]
        group1_ids = {t.id for t in plan[1]}
        assert group1_ids == {"b", "c"}
        assert [t.id for t in plan[2]] == ["d"]

    def test_independent_tasks_in_single_group(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        scheduler.add_task(Task(id="b", name="B"), _noop_coro())
        scheduler.add_task(Task(id="c", name="C"), _noop_coro())

        plan = scheduler.get_execution_plan()
        assert len(plan) == 1
        assert {t.id for t in plan[0]} == {"a", "b", "c"}

    def test_priority_ordering_within_group(self):
        """Higher-priority tasks come first within the same group."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="lo", name="Low", priority=1), _noop_coro())
        scheduler.add_task(Task(id="hi", name="High", priority=10), _noop_coro())
        scheduler.add_task(Task(id="mid", name="Mid", priority=5), _noop_coro())

        plan = scheduler.get_execution_plan()
        ids = [t.id for t in plan[0]]
        assert ids == ["hi", "mid", "lo"]

    def test_unknown_dependency_raises(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", dependencies=["ghost"]),
            _noop_coro(),
        )
        with pytest.raises(KeyError, match="unknown task 'ghost'"):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_tasks_execute_in_dependency_order(self):
        """Verify execution order respects dependencies at runtime."""
        order: list[str] = []

        async def make_coro(task_id: str):
            async def _fn():
                order.append(task_id)
            return _fn

        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _make_order_coro(order, "a"))
        scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _make_order_coro(order, "b"))
        scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _make_order_coro(order, "c"))

        await scheduler.run()

        assert order == ["a", "b", "c"]


def _make_order_coro(order_list: list[str], task_id: str):
    """Helper: returns an async fn that appends task_id to order_list."""
    async def _fn():
        order_list.append(task_id)
    return _fn


# ================================================================== #
#  5. Circular dependency detection
# ================================================================== #

class TestCircularDependency:
    def test_self_loop(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", dependencies=["a"]), _noop_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", dependencies=["b"]), _noop_coro())
        scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A", dependencies=["c"]), _noop_coro())
        scheduler.add_task(Task(id="b", name="B", dependencies=["a"]), _noop_coro())
        scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _noop_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_circular_dependency_prevents_run(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="x", name="X", dependencies=["y"]), _noop_coro())
        scheduler.add_task(Task(id="y", name="Y", dependencies=["x"]), _noop_coro())

        with pytest.raises(CircularDependencyError):
            await scheduler.run()

    def test_partial_cycle_in_larger_graph(self):
        """Only B and C form a cycle; A is independent. Should still detect."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        scheduler.add_task(Task(id="b", name="B", dependencies=["c"]), _noop_coro())
        scheduler.add_task(Task(id="c", name="C", dependencies=["b"]), _noop_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()


# ================================================================== #
#  6. Retry logic with exponential backoff
# ================================================================== #

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_succeeds_after_retries(self):
        scheduler = TaskScheduler()
        fn = _fail_n_then_succeed(2, result="ok")
        task = Task(id="r", name="Retry", max_retries=3)
        scheduler.add_task(task, fn)

        metrics = await scheduler.run()

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "ok"
        assert task.retry_count == 2
        assert metrics.retry_counts["r"] == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self):
        scheduler = TaskScheduler()
        task = Task(id="f", name="Fail", max_retries=2)
        scheduler.add_task(task, _fail_coro(RuntimeError("permanent")))

        await scheduler.run()

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2
        assert task.result is None

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        scheduler = TaskScheduler()
        task = Task(id="z", name="NoRetry", max_retries=0)
        scheduler.add_task(task, _fail_coro(ValueError("boom")))

        await scheduler.run()

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 0

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify backoff delays are approximately 0.1s, 0.2s, 0.4s."""
        scheduler = TaskScheduler()
        task = Task(id="bo", name="Backoff", max_retries=3)
        scheduler.add_task(task, _fail_coro(RuntimeError("fail")))

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # Expected minimum: 0.1 + 0.2 + 0.4 = 0.7s (3 retries, sleep before each)
        # Allow generous tolerance for CI variability.
        assert elapsed >= 0.6
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_in_metrics(self):
        scheduler = TaskScheduler()
        fn = _fail_n_then_succeed(1, result="recovered")
        task = Task(id="m", name="Metrics", max_retries=3)
        scheduler.add_task(task, fn)

        metrics = await scheduler.run()

        assert metrics.retry_counts["m"] == 1
        assert metrics.per_task_time["m"] > 0


# ================================================================== #
#  7. Concurrent execution respecting concurrency limits
# ================================================================== #

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """No more than max_concurrency tasks should run simultaneously."""
        max_concurrent = 2
        scheduler = TaskScheduler(max_concurrency=max_concurrent)

        peak_concurrency = {"value": 0}
        current_concurrency = {"value": 0}

        def make_tracked_coro():
            async def _fn():
                current_concurrency["value"] += 1
                if current_concurrency["value"] > peak_concurrency["value"]:
                    peak_concurrency["value"] = current_concurrency["value"]
                await asyncio.sleep(0.05)
                current_concurrency["value"] -= 1
            return _fn

        for i in range(6):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"Task {i}"),
                make_tracked_coro(),
            )

        await scheduler.run()

        assert peak_concurrency["value"] <= max_concurrent

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_low_concurrency(self):
        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(4):
            scheduler.add_task(
                Task(id=f"s{i}", name=f"Serial {i}"),
                _result_coro(i),
            )

        await scheduler.run()

        for i in range(4):
            assert scheduler.tasks[f"s{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_concurrency_with_dependency_layers(self):
        """Two layers: layer-0 has 4 tasks (concurrency=2), layer-1 depends on all."""
        scheduler = TaskScheduler(max_concurrency=2)

        peak = {"value": 0}
        current = {"value": 0}

        def tracked():
            async def _fn():
                current["value"] += 1
                if current["value"] > peak["value"]:
                    peak["value"] = current["value"]
                await asyncio.sleep(0.05)
                current["value"] -= 1
            return _fn

        for i in range(4):
            scheduler.add_task(Task(id=f"l0_{i}", name=f"L0-{i}"), tracked())

        scheduler.add_task(
            Task(id="l1", name="L1", dependencies=[f"l0_{i}" for i in range(4)]),
            tracked(),
        )

        await scheduler.run()

        assert peak["value"] <= 2
        assert scheduler.tasks["l1"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_tasks_faster_than_serial(self):
        """With concurrency=4, four 0.1s tasks should finish in ~0.1s, not ~0.4s."""
        scheduler = TaskScheduler(max_concurrency=4)

        async def slow():
            await asyncio.sleep(0.1)

        for i in range(4):
            scheduler.add_task(Task(id=f"p{i}", name=f"P{i}"), slow)

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # Should be around 0.1s, well under 0.35s
        assert elapsed < 0.35


# ================================================================== #
#  8. Observer / Event system
# ================================================================== #

class TestObserverEvents:
    @pytest.mark.asyncio
    async def test_on_task_start_callback(self):
        started: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())

        await scheduler.run()

        assert started == ["a"]

    @pytest.mark.asyncio
    async def test_on_task_complete_callback(self):
        completed: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: completed.append(t.id))
        scheduler.add_task(Task(id="a", name="A"), _result_coro(1))

        await scheduler.run()

        assert completed == ["a"]

    @pytest.mark.asyncio
    async def test_on_task_fail_callback(self):
        failures: list[tuple[str, str]] = []
        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, exc: failures.append((t.id, str(exc))))
        scheduler.add_task(
            Task(id="f", name="Fail", max_retries=0),
            _fail_coro(RuntimeError("oops")),
        )

        await scheduler.run()

        assert len(failures) == 1
        assert failures[0][0] == "f"
        assert "oops" in failures[0][1]

    @pytest.mark.asyncio
    async def test_multiple_listeners_on_same_event(self):
        calls_a: list[str] = []
        calls_b: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_complete(lambda t: calls_a.append(t.id))
        scheduler.on_task_complete(lambda t: calls_b.append(t.id))
        scheduler.add_task(Task(id="m", name="M"), _noop_coro())

        await scheduler.run()

        assert calls_a == ["m"]
        assert calls_b == ["m"]

    @pytest.mark.asyncio
    async def test_async_listener(self):
        results: list[str] = []

        async def async_callback(task: Task):
            results.append(task.id)

        scheduler = TaskScheduler()
        scheduler.on_task_complete(async_callback)
        scheduler.add_task(Task(id="ac", name="Async CB"), _noop_coro())

        await scheduler.run()

        assert results == ["ac"]

    @pytest.mark.asyncio
    async def test_all_events_fire_for_successful_task(self):
        events: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: events.append(f"start:{t.id}"))
        scheduler.on_task_complete(lambda t: events.append(f"complete:{t.id}"))
        scheduler.add_task(Task(id="e", name="E"), _noop_coro())

        await scheduler.run()

        assert events == ["start:e", "complete:e"]


# ================================================================== #
#  9. Failure propagation to downstream tasks
# ================================================================== #

class TestFailurePropagation:
    @pytest.mark.asyncio
    async def test_downstream_task_skipped_when_dependency_fails(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", max_retries=0),
            _fail_coro(RuntimeError("boom")),
        )
        scheduler.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _result_coro("never"),
        )

        await scheduler.run()

        assert scheduler.tasks["a"].status == TaskStatus.FAILED
        assert scheduler.tasks["b"].status == TaskStatus.FAILED
        assert scheduler.tasks["b"].result is None

    @pytest.mark.asyncio
    async def test_fail_callback_fires_for_skipped_downstream(self):
        fail_ids: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, exc: fail_ids.append(t.id))

        scheduler.add_task(
            Task(id="a", name="A", max_retries=0),
            _fail_coro(RuntimeError("x")),
        )
        scheduler.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _noop_coro(),
        )

        await scheduler.run()

        assert "a" in fail_ids
        assert "b" in fail_ids

    @pytest.mark.asyncio
    async def test_sibling_unaffected_by_failure(self):
        """If A fails, B (independent) should still succeed."""
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", max_retries=0),
            _fail_coro(RuntimeError("fail")),
        )
        scheduler.add_task(Task(id="b", name="B"), _result_coro(99))

        await scheduler.run()

        assert scheduler.tasks["a"].status == TaskStatus.FAILED
        assert scheduler.tasks["b"].status == TaskStatus.COMPLETED
        assert scheduler.tasks["b"].result == 99


# ================================================================== #
# 10. Metrics
# ================================================================== #

class TestMetrics:
    @pytest.mark.asyncio
    async def test_total_time_recorded(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        metrics = await scheduler.run()
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_per_task_time_recorded(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        scheduler.add_task(Task(id="b", name="B"), _noop_coro())
        metrics = await scheduler.run()
        assert "a" in metrics.per_task_time
        assert "b" in metrics.per_task_time

    @pytest.mark.asyncio
    async def test_metrics_property_matches_run_return(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="A"), _noop_coro())
        metrics = await scheduler.run()
        assert scheduler.metrics is metrics

    @pytest.mark.asyncio
    async def test_failed_task_has_metrics(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="f", name="Fail", max_retries=0),
            _fail_coro(RuntimeError("x")),
        )
        metrics = await scheduler.run()
        assert "f" in metrics.per_task_time
        assert metrics.retry_counts["f"] == 0

    @pytest.mark.asyncio
    async def test_skipped_task_has_zero_time(self):
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="A", max_retries=0),
            _fail_coro(RuntimeError("x")),
        )
        scheduler.add_task(
            Task(id="b", name="B", dependencies=["a"]),
            _noop_coro(),
        )
        metrics = await scheduler.run()
        assert metrics.per_task_time["b"] == 0.0
        assert metrics.retry_counts["b"] == 0


# ================================================================== #
# 11. Empty scheduler
# ================================================================== #

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_without_error(self):
        scheduler = TaskScheduler()
        metrics = await scheduler.run()
        assert metrics.total_time >= 0
        assert metrics.per_task_time == {}

    def test_execution_plan_empty(self):
        scheduler = TaskScheduler()
        plan = scheduler.get_execution_plan()
        assert plan == []
