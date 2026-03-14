"""
test_task_scheduler.py

Pytest test suite for task_scheduler.py.

Coverage:
  1. Basic task execution
  2. Dependency resolution
  3. Circular dependency detection
  4. Retry logic with exponential backoff
  5. Concurrent execution respecting concurrency limits
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

# Helper to run coroutines in tests (Python 3.10+ compatible)
_run = asyncio.run

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(
    task_id: str,
    *,
    name: str | None = None,
    priority: int = 5,
    dependencies: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=task_id,
        name=name or task_id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


def simple_coro(value: object = None):
    """Return an async callable that resolves to *value*."""

    async def _coro():
        return value

    return _coro


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_single_task_completes(self):
        """A single task with no dependencies should complete and return its result."""
        scheduler = TaskScheduler()
        task = make_task("t1")
        scheduler.add_task(task, simple_coro("hello"))

        results = asyncio.run(scheduler.run())

        assert results["t1"] == "hello"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "hello"

    def test_multiple_independent_tasks_all_complete(self):
        """All tasks with no dependencies should complete."""
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.add_task(make_task(f"t{i}"), simple_coro(i * 10))

        results = asyncio.run(scheduler.run())

        assert len(results) == 5
        for i in range(5):
            assert results[f"t{i}"] == i * 10

    def test_observer_on_task_start_and_complete_fired(self):
        """on_task_start and on_task_complete callbacks must be invoked."""
        scheduler = TaskScheduler()
        started: list[str] = []
        completed: list[str] = []

        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.on_task_complete(lambda t: completed.append(t.id))

        scheduler.add_task(make_task("t1"), simple_coro())
        asyncio.run(scheduler.run())

        assert "t1" in started
        assert "t1" in completed

    def test_metrics_populated_after_run(self):
        """Scheduler metrics should contain timing for each task after run()."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("t1"), simple_coro(42))
        asyncio.run(scheduler.run())

        m = scheduler.metrics
        assert m is not None
        assert m.total_elapsed is not None and m.total_elapsed >= 0
        assert "t1" in m.tasks
        assert m.tasks["t1"].elapsed is not None and m.tasks["t1"].elapsed >= 0

    def test_duplicate_task_id_raises(self):
        """Registering two tasks with the same ID should raise ValueError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("dup"), simple_coro())
        with pytest.raises(ValueError, match="already registered"):
            scheduler.add_task(make_task("dup"), simple_coro())


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------


class TestDependencyResolution:
    def test_dependent_task_receives_predecessor_result(self):
        """A task should only run after all its dependencies have completed."""
        execution_order: list[str] = []

        async def coro_a():
            execution_order.append("a")
            return "A"

        async def coro_b():
            execution_order.append("b")
            return "B"

        scheduler = TaskScheduler()
        task_a = make_task("a")
        task_b = make_task("b", dependencies=["a"])
        scheduler.add_task(task_a, coro_a)
        scheduler.add_task(task_b, coro_b)

        asyncio.run(scheduler.run())

        assert execution_order.index("a") < execution_order.index("b")
        assert task_a.status == TaskStatus.COMPLETED
        assert task_b.status == TaskStatus.COMPLETED

    def test_get_execution_plan_groups(self):
        """get_execution_plan should group independent tasks together."""
        scheduler = TaskScheduler()
        # a and b are independent; c depends on both
        scheduler.add_task(make_task("a"), simple_coro())
        scheduler.add_task(make_task("b"), simple_coro())
        scheduler.add_task(make_task("c", dependencies=["a", "b"]), simple_coro())

        plan = scheduler.get_execution_plan()

        # First group must contain a and b; second group must contain c
        assert len(plan) == 2
        first_group = set(plan[0])
        second_group = set(plan[1])
        assert {"a", "b"} == first_group
        assert {"c"} == second_group

    def test_deep_chain_executes_in_order(self):
        """A → B → C → D should execute strictly in sequence."""
        order: list[str] = []

        def make_coro(tid: str):
            async def _c():
                order.append(tid)

            return _c

        scheduler = TaskScheduler()
        scheduler.add_task(make_task("A"), make_coro("A"))
        scheduler.add_task(make_task("B", dependencies=["A"]), make_coro("B"))
        scheduler.add_task(make_task("C", dependencies=["B"]), make_coro("C"))
        scheduler.add_task(make_task("D", dependencies=["C"]), make_coro("D"))

        asyncio.run(scheduler.run())

        assert order == ["A", "B", "C", "D"]

    def test_priority_within_group(self):
        """Within a single execution group, higher-priority tasks should appear first
        in the execution plan."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("low", priority=1), simple_coro())
        scheduler.add_task(make_task("high", priority=9), simple_coro())
        scheduler.add_task(make_task("mid", priority=5), simple_coro())

        plan = scheduler.get_execution_plan()

        assert len(plan) == 1
        assert plan[0][0] == "high"
        assert plan[0][-1] == "low"


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDependencyDetection:
    def test_simple_cycle_raises(self):
        """A ↔ B forms a direct cycle and must raise CircularDependencyError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["b"]), simple_coro())
        scheduler.add_task(make_task("b", dependencies=["a"]), simple_coro())

        with pytest.raises(CircularDependencyError):
            asyncio.run(scheduler.run())

    def test_three_node_cycle_raises(self):
        """A → B → C → A forms a three-node cycle."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("a", dependencies=["c"]), simple_coro())
        scheduler.add_task(make_task("b", dependencies=["a"]), simple_coro())
        scheduler.add_task(make_task("c", dependencies=["b"]), simple_coro())

        with pytest.raises(CircularDependencyError):
            asyncio.run(scheduler.run())

    def test_get_execution_plan_detects_cycle(self):
        """get_execution_plan should also raise CircularDependencyError."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("x", dependencies=["y"]), simple_coro())
        scheduler.add_task(make_task("y", dependencies=["x"]), simple_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_self_dependency_raises(self):
        """A task depending on itself is a trivial cycle."""
        scheduler = TaskScheduler()
        scheduler.add_task(make_task("self", dependencies=["self"]), simple_coro())

        with pytest.raises(CircularDependencyError):
            asyncio.run(scheduler.run())

    def test_dag_without_cycle_does_not_raise(self):
        """A valid DAG (diamond shape) must not raise."""
        scheduler = TaskScheduler()
        #     root
        #    /    \
        #  left  right
        #    \    /
        #     tip
        scheduler.add_task(make_task("root"), simple_coro())
        scheduler.add_task(make_task("left", dependencies=["root"]), simple_coro())
        scheduler.add_task(make_task("right", dependencies=["root"]), simple_coro())
        scheduler.add_task(
            make_task("tip", dependencies=["left", "right"]), simple_coro()
        )

        plan = scheduler.get_execution_plan()  # must not raise
        assert len(plan) == 3


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_task_retries_on_failure_then_succeeds(self):
        """A task that fails twice then succeeds should end in COMPLETED."""
        attempts = {"count": 0}

        async def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient error")
            return "ok"

        scheduler = TaskScheduler()
        task = make_task("flaky", max_retries=3)
        scheduler.add_task(task, flaky)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "ok"
        assert task.retry_count == 2  # two retries before success

    def test_task_marked_failed_after_all_retries_exhausted(self):
        """A task that always raises should end in FAILED after max_retries."""

        async def always_fails():
            raise ValueError("permanent error")

        scheduler = TaskScheduler()
        task = make_task("bad", max_retries=2)
        scheduler.add_task(task, always_fails)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2  # retried max_retries times

    def test_on_task_fail_callback_fired(self):
        """on_task_fail callback must be called when a task exhausts retries."""
        failed_tasks: list[str] = []

        async def always_fails():
            raise RuntimeError("boom")

        scheduler = TaskScheduler()
        scheduler.on_task_fail(lambda t, exc: failed_tasks.append(t.id))
        task = make_task("f", max_retries=1)
        scheduler.add_task(task, always_fails)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(scheduler.run())

        assert "f" in failed_tasks

    def test_exponential_backoff_delays(self):
        """asyncio.sleep should be called with exponentially increasing delays."""

        async def always_fails():
            raise RuntimeError("err")

        scheduler = TaskScheduler()
        task = make_task("exp", max_retries=3)
        scheduler.add_task(task, always_fails)

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(scheduler.run())

        # Expected: 1.0, 2.0, 4.0  (base=1, exponent=attempt-1)
        assert sleep_calls == pytest.approx([1.0, 2.0, 4.0])

    def test_no_retries_when_max_retries_zero(self):
        """With max_retries=0 a failing task should fail immediately, no sleep."""

        async def always_fails():
            raise RuntimeError("instant fail")

        scheduler = TaskScheduler()
        task = make_task("z", max_retries=0)
        scheduler.add_task(task, always_fails)

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.run(scheduler.run())

        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 0
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    def test_concurrency_limit_not_exceeded(self):
        """Never more than *limit* tasks should run simultaneously."""
        limit = 2
        active = {"count": 0, "max_observed": 0}

        async def tracked_coro():
            active["count"] += 1
            active["max_observed"] = max(active["max_observed"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1

        scheduler = TaskScheduler(concurrency_limit=limit)
        for i in range(6):
            scheduler.add_task(make_task(f"t{i}"), tracked_coro)

        asyncio.run(scheduler.run())

        assert active["max_observed"] <= limit

    def test_concurrency_limit_one_is_serial(self):
        """With concurrency_limit=1 tasks must run one at a time (serial)."""
        active = {"count": 0, "violated": False}

        async def serial_coro():
            if active["count"] > 0:
                active["violated"] = True
            active["count"] += 1
            await asyncio.sleep(0.01)
            active["count"] -= 1

        scheduler = TaskScheduler(concurrency_limit=1)
        for i in range(4):
            scheduler.add_task(make_task(f"t{i}"), serial_coro)

        asyncio.run(scheduler.run())

        assert not active["violated"], "More than 1 task ran concurrently"

    def test_higher_concurrency_limit_improves_throughput(self):
        """Running 6 tasks with limit=6 should be faster than with limit=1."""
        task_delay = 0.05

        async def slow_coro():
            await asyncio.sleep(task_delay)

        def run_with_limit(lim: int) -> float:
            sched = TaskScheduler(concurrency_limit=lim)
            for i in range(6):
                sched.add_task(make_task(f"t{i}"), slow_coro)
            start = time.monotonic()
            asyncio.run(sched.run())
            return time.monotonic() - start

        serial_time = run_with_limit(1)
        parallel_time = run_with_limit(6)

        assert parallel_time < serial_time, (
            f"Expected parallel ({parallel_time:.2f}s) < serial ({serial_time:.2f}s)"
        )

    def test_invalid_concurrency_limit_raises(self):
        """concurrency_limit < 1 should raise ValueError at construction time."""
        with pytest.raises(ValueError):
            TaskScheduler(concurrency_limit=0)

    def test_all_tasks_complete_regardless_of_limit(self):
        """Every task must eventually complete even with tight concurrency."""
        scheduler = TaskScheduler(concurrency_limit=2)
        for i in range(10):
            scheduler.add_task(make_task(f"t{i}"), simple_coro(i))

        results = asyncio.run(scheduler.run())

        assert len(results) == 10
        for i in range(10):
            assert results[f"t{i}"] == i
