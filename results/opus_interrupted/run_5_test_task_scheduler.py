"""Tests for the async task scheduler."""

from __future__ import annotations

import asyncio
import time

import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_coro(result: object = "ok"):
    """Return a coroutine factory that succeeds immediately."""
    async def _run(task: Task) -> object:
        return result
    return _run


def _failing_coro(fail_times: int = 1):
    """Return a coroutine factory that fails *fail_times* before succeeding."""
    call_count = {"n": 0}

    async def _run(task: Task) -> str:
        call_count["n"] += 1
        if call_count["n"] <= fail_times:
            raise RuntimeError(f"Transient failure #{call_count['n']}")
        return "recovered"
    return _run


def _always_failing_coro():
    """Return a coroutine factory that always raises."""
    async def _run(task: Task) -> None:
        raise RuntimeError("permanent failure")
    return _run


def _slow_coro(duration: float = 0.05):
    """Return a coroutine factory that sleeps for *duration* seconds."""
    async def _run(task: Task) -> str:
        await asyncio.sleep(duration)
        return "done"
    return _run


# ---------------------------------------------------------------------------
# 1. Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    """Verify that single and multiple independent tasks execute correctly."""

    @pytest.mark.asyncio
    async def test_single_task_completes(self) -> None:
        scheduler = TaskScheduler()
        t = Task(id="a", name="alpha")
        scheduler.add_task(t, _simple_coro("hello"))

        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "hello"
        assert metrics.total_time > 0
        assert "a" in metrics.per_task_time

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks(self) -> None:
        scheduler = TaskScheduler()
        ids = ["a", "b", "c"]
        for tid in ids:
            scheduler.add_task(Task(id=tid, name=tid), _simple_coro(tid))

        await scheduler.run()

        for tid in ids:
            task = scheduler.tasks[tid]
            assert task.status == TaskStatus.COMPLETED
            assert task.result == tid

    @pytest.mark.asyncio
    async def test_observer_callbacks_fire(self) -> None:
        started: list[str] = []
        completed: list[str] = []
        scheduler = TaskScheduler()
        scheduler.on_task_start(lambda t: started.append(t.id))
        scheduler.on_task_complete(lambda t: completed.append(t.id))

        scheduler.add_task(Task(id="x", name="x"), _simple_coro())
        await scheduler.run()

        assert "x" in started
        assert "x" in completed


# ---------------------------------------------------------------------------
# 2. Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    """Verify that tasks execute in correct dependency order."""

    @pytest.mark.asyncio
    async def test_linear_chain(self) -> None:
        """a -> b -> c: must run in three sequential groups."""
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a"), _simple_coro())
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), _simple_coro())
        scheduler.add_task(Task(id="c", name="c", dependencies=["b"]), _simple_coro())

        plan = scheduler.get_execution_plan()

        # Flatten the plan to verify ordering constraints.
        flat: list[str] = [tid for group in plan for tid in group]
        assert flat.index("a") < flat.index("b") < flat.index("c")

    @pytest.mark.asyncio
    async def test_diamond_dependency(self) -> None:
        """
        a -> b, a -> c, b -> d, c -> d
        d must appear after both b and c; b and c can be parallel.
        """
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a"), _simple_coro())
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), _simple_coro())
        scheduler.add_task(Task(id="c", name="c", dependencies=["a"]), _simple_coro())
        scheduler.add_task(
            Task(id="d", name="d", dependencies=["b", "c"]), _simple_coro()
        )

        plan = scheduler.get_execution_plan()

        assert plan[0] == ["a"]
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    @pytest.mark.asyncio
    async def test_execution_respects_dependency_order(self) -> None:
        """Run the diamond and verify all complete."""
        scheduler = TaskScheduler()
        order: list[str] = []

        async def track(task: Task) -> None:
            order.append(task.id)

        scheduler.add_task(Task(id="a", name="a"), track)
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), track)
        scheduler.add_task(Task(id="c", name="c", dependencies=["a"]), track)
        scheduler.add_task(
            Task(id="d", name="d", dependencies=["b", "c"]), track
        )

        await scheduler.run()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")


# ---------------------------------------------------------------------------
# 3. Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependency:
    """Verify that cycles in the dependency graph raise CircularDependencyError."""

    def test_self_loop(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a", dependencies=["a"]), _simple_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_two_node_cycle(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a", dependencies=["b"]), _simple_coro())
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), _simple_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    def test_three_node_cycle(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a", dependencies=["c"]), _simple_coro())
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), _simple_coro())
        scheduler.add_task(Task(id="c", name="c", dependencies=["b"]), _simple_coro())

        with pytest.raises(CircularDependencyError):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_cycle_prevents_run(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a", dependencies=["b"]), _simple_coro())
        scheduler.add_task(Task(id="b", name="b", dependencies=["a"]), _simple_coro())

        with pytest.raises(CircularDependencyError):
            await scheduler.run()


# ---------------------------------------------------------------------------
# 4. Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Verify exponential backoff retries on transient failures."""

    @pytest.mark.asyncio
    async def test_recovers_after_transient_failure(self) -> None:
        scheduler = TaskScheduler(base_backoff=0.01)
        t = Task(id="r", name="retry-me", max_retries=3)
        scheduler.add_task(t, _failing_coro(fail_times=2))

        metrics = await scheduler.run()

        assert t.status == TaskStatus.COMPLETED
        assert t.result == "recovered"
        assert t.retry_count == 2
        assert metrics.retry_counts["r"] == 2

    @pytest.mark.asyncio
    async def test_permanent_failure_exhausts_retries(self) -> None:
        failed_tasks: list[str] = []
        scheduler = TaskScheduler(base_backoff=0.01)
        scheduler.on_task_fail(lambda t, e: failed_tasks.append(t.id))

        t = Task(id="f", name="fail-me", max_retries=2)
        scheduler.add_task(t, _always_failing_coro())

        await scheduler.run()

        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 3  # initial + 2 retries = 3 attempts total
        assert "f" in failed_tasks

    @pytest.mark.asyncio
    async def test_backoff_delay_increases(self) -> None:
        """Verify that retry delay grows exponentially."""
        scheduler = TaskScheduler(base_backoff=0.05)
        t = Task(id="b", name="backoff", max_retries=3)
        scheduler.add_task(t, _failing_coro(fail_times=2))

        start = time.monotonic()
        await scheduler.run()
        elapsed = time.monotonic() - start

        # With base=0.05:  delay1 = 0.05, delay2 = 0.10  -> total >= 0.15
        assert t.status == TaskStatus.COMPLETED
        assert elapsed >= 0.12  # small margin for scheduling jitter


# ---------------------------------------------------------------------------
# 5. Concurrent execution with concurrency limit
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    """Verify that the scheduler respects the max_concurrency setting."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        max_concurrent = 2
        scheduler = TaskScheduler(max_concurrency=max_concurrent)

        running_count: list[int] = []
        peak = {"value": 0}
        lock = asyncio.Lock()

        async def instrumented(task: Task) -> str:
            async with lock:
                running_count.append(1)
                current = len(running_count)
                if current > peak["value"]:
                    peak["value"] = current
            await asyncio.sleep(0.05)
            async with lock:
                running_count.pop()
            return "ok"

        for i in range(6):
            scheduler.add_task(
                Task(id=f"t{i}", name=f"task-{i}"), instrumented
            )

        await scheduler.run()

        assert peak["value"] <= max_concurrent

    @pytest.mark.asyncio
    async def test_all_tasks_complete_with_limit(self) -> None:
        scheduler = TaskScheduler(max_concurrency=1)
        for i in range(4):
            scheduler.add_task(
                Task(id=f"s{i}", name=f"seq-{i}"), _simple_coro(i)
            )

        await scheduler.run()

        for i in range(4):
            assert scheduler.tasks[f"s{i}"].status == TaskStatus.COMPLETED
            assert scheduler.tasks[f"s{i}"].result == i


# ---------------------------------------------------------------------------
# Edge cases / additional coverage
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional tests for edge cases and validation."""

    def test_priority_validation(self) -> None:
        with pytest.raises(ValueError):
            Task(id="bad", name="bad", priority=0)
        with pytest.raises(ValueError):
            Task(id="bad", name="bad", priority=11)

    def test_duplicate_task_id_rejected(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="a", name="a"), _simple_coro())
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(Task(id="a", name="a2"), _simple_coro())

    def test_unknown_dependency_raises(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(
            Task(id="a", name="a", dependencies=["nonexistent"]), _simple_coro()
        )
        with pytest.raises(ValueError, match="unknown task"):
            scheduler.get_execution_plan()

    @pytest.mark.asyncio
    async def test_empty_scheduler(self) -> None:
        scheduler = TaskScheduler()
        metrics = await scheduler.run()
        assert metrics.total_time >= 0
        assert metrics.per_task_time == {}

    @pytest.mark.asyncio
    async def test_metrics_populated(self) -> None:
        scheduler = TaskScheduler()
        scheduler.add_task(Task(id="m", name="metrics"), _simple_coro())
        metrics = await scheduler.run()

        assert "m" in metrics.per_task_time
        assert metrics.per_task_time["m"] >= 0
        assert metrics.retry_counts["m"] == 0
        assert metrics.total_time > 0
