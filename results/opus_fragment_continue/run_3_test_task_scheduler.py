"""Tests for the async task scheduler module."""

from __future__ import annotations

import asyncio
import pytest

from task_scheduler import (
    CircularDependencyError,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────


async def noop_coro(task: Task) -> str:
    """A trivial coroutine that always succeeds."""
    return f"{task.id}-done"


async def slow_coro(task: Task) -> str:
    """A coroutine that sleeps briefly."""
    await asyncio.sleep(0.05)
    return f"{task.id}-slow"


async def failing_coro(task: Task) -> str:
    """A coroutine that always raises."""
    raise RuntimeError(f"{task.id} exploded")


def _make_failing_after(n: int):
    """Return a coroutine factory that fails the first *n* calls then succeeds."""
    call_count = {"value": 0}

    async def coro(task: Task) -> str:
        call_count["value"] += 1
        if call_count["value"] <= n:
            raise RuntimeError("transient failure")
        return f"{task.id}-recovered"

    return coro


# ═══════════════════════════════════════════════════════════════════
# Task dataclass
# ═══════════════════════════════════════════════════════════════════


class TestTaskDataclass:
    """Tests for the Task dataclass itself."""

    def test_default_values(self) -> None:
        t = Task(id="t1", name="Test")
        assert t.priority == 5
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 3
        assert t.result is None

    def test_priority_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="T", priority=0)

    def test_priority_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="Priority must be between 1 and 10"):
            Task(id="t", name="T", priority=11)

    def test_valid_priority_edges(self) -> None:
        t_low = Task(id="t1", name="Low", priority=1)
        t_high = Task(id="t2", name="High", priority=10)
        assert t_low.priority == 1
        assert t_high.priority == 10

    def test_created_at_is_set(self) -> None:
        t = Task(id="t", name="T")
        assert t.created_at is not None
        assert t.created_at.tzinfo is not None  # should be UTC-aware


# ═══════════════════════════════════════════════════════════════════
# TaskMetrics
# ═══════════════════════════════════════════════════════════════════


class TestTaskMetrics:
    def test_defaults(self) -> None:
        m = TaskMetrics()
        assert m.total_time == 0.0
        assert m.per_task_time == {}
        assert m.retry_counts == {}


# ═══════════════════════════════════════════════════════════════════
# TaskScheduler – construction
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerConstruction:
    def test_default_concurrency(self) -> None:
        s = TaskScheduler()
        assert s.max_concurrency == 4

    def test_custom_concurrency(self) -> None:
        s = TaskScheduler(max_concurrency=8)
        assert s.max_concurrency == 8

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskScheduler(max_concurrency=0)


# ═══════════════════════════════════════════════════════════════════
# TaskScheduler – task management
# ═══════════════════════════════════════════════════════════════════


class TestTaskManagement:
    def test_add_and_get(self) -> None:
        s = TaskScheduler()
        t = Task(id="a", name="A")
        s.add_task(t, noop_coro)
        assert s.get_task("a") is t

    def test_add_duplicate_raises(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(Task(id="a", name="A2"), noop_coro)

    def test_get_unknown_raises(self) -> None:
        s = TaskScheduler()
        with pytest.raises(KeyError):
            s.get_task("nonexistent")

    def test_tasks_property_sorted_by_priority(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="low", name="Low", priority=1), noop_coro)
        s.add_task(Task(id="high", name="High", priority=10), noop_coro)
        s.add_task(Task(id="mid", name="Mid", priority=5), noop_coro)
        ids = [t.id for t in s.tasks]
        assert ids == ["high", "mid", "low"]


# ═══════════════════════════════════════════════════════════════════
# Dependency validation
# ═══════════════════════════════════════════════════════════════════


class TestDependencyValidation:
    @pytest.mark.asyncio
    async def test_unknown_dependency_raises(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["ghost"]), noop_coro)
        with pytest.raises(KeyError, match="unknown task 'ghost'"):
            await s.run()


# ═══════════════════════════════════════════════════════════════════
# Topological sort / circular dependency detection
# ═══════════════════════════════════════════════════════════════════


class TestTopologicalSort:
    def test_linear_chain(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), noop_coro)
        groups = s._topological_sort()
        assert len(groups) == 3
        assert groups[0] == ["a"]
        assert groups[1] == ["b"]
        assert groups[2] == ["c"]

    def test_independent_tasks_same_group(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B"), noop_coro)
        groups = s._topological_sort()
        assert len(groups) == 1
        assert set(groups[0]) == {"a", "b"}

    def test_diamond_dependency(self) -> None:
        """A -> B, A -> C, B -> D, C -> D."""
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="d", name="D", dependencies=["b", "c"]), noop_coro)
        groups = s._topological_sort()
        assert groups[0] == ["a"]
        assert set(groups[1]) == {"b", "c"}
        assert groups[2] == ["d"]

    def test_circular_dependency_detected(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["b"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        with pytest.raises(CircularDependencyError):
            s._topological_sort()

    def test_circular_dependency_three_nodes(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", dependencies=["c"]), noop_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), noop_coro)
        with pytest.raises(CircularDependencyError):
            s._topological_sort()

    def test_circular_dependency_error_has_cycle(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="x", name="X", dependencies=["y"]), noop_coro)
        s.add_task(Task(id="y", name="Y", dependencies=["x"]), noop_coro)
        with pytest.raises(CircularDependencyError) as exc_info:
            s._topological_sort()
        assert len(exc_info.value.cycle) >= 2

    def test_priority_ordering_within_group(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="low", name="Low", priority=1), noop_coro)
        s.add_task(Task(id="high", name="High", priority=10), noop_coro)
        groups = s._topological_sort()
        assert groups[0][0] == "high"
        assert groups[0][1] == "low"


# ═══════════════════════════════════════════════════════════════════
# Execution – basic runs
# ═══════════════════════════════════════════════════════════════════


class TestExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        metrics = await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "a-done"
        assert "a" in metrics.per_task_time
        assert metrics.total_time > 0

    @pytest.mark.asyncio
    async def test_chain_execution_order(self) -> None:
        execution_order: list[str] = []

        async def tracking_coro(task: Task) -> str:
            execution_order.append(task.id)
            return task.id

        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), tracking_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), tracking_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), tracking_coro)
        await s.run()

        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_concurrent_independent_tasks(self) -> None:
        """Independent tasks should run concurrently, not sequentially."""
        timestamps: dict[str, float] = {}

        async def timed_coro(task: Task) -> str:
            import time as _t

            timestamps[f"{task.id}_start"] = _t.monotonic()
            await asyncio.sleep(0.1)
            timestamps[f"{task.id}_end"] = _t.monotonic()
            return task.id

        s = TaskScheduler(max_concurrency=4)
        for i in range(4):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), timed_coro)
        await s.run()

        # All 4 should start before any finishes (concurrent)
        starts = [timestamps[f"t{i}_start"] for i in range(4)]
        ends = [timestamps[f"t{i}_end"] for i in range(4)]
        assert max(starts) < min(ends), "Tasks did not run concurrently"

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        """No more than max_concurrency tasks run at the same time."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_coro(task: Task) -> str:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return task.id

        s = TaskScheduler(max_concurrency=2)
        for i in range(6):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), counting_coro)
        await s.run()

        assert max_concurrent <= 2


# ═══════════════════════════════════════════════════════════════════
# Retry logic
# ═══════════════════════════════════════════════════════════════════


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure(self) -> None:
        coro = _make_failing_after(2)  # fails twice, then succeeds
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        metrics = await s.run()

        task = s.get_task("a")
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 2
        assert task.result == "a-recovered"
        assert metrics.retry_counts["a"] == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=2), failing_coro)
        await s.run()

        task = s.get_task("a")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 3  # initial + 2 retries = 3 total attempts

    @pytest.mark.asyncio
    async def test_zero_max_retries(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), failing_coro)
        await s.run()

        task = s.get_task("a")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 1


# ═══════════════════════════════════════════════════════════════════
# Failure propagation – cancel dependents
# ═══════════════════════════════════════════════════════════════════


class TestFailurePropagation:
    @pytest.mark.asyncio
    async def test_dependents_cancelled_on_failure(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), failing_coro)
        s.add_task(Task(id="b", name="B", dependencies=["a"]), noop_coro)
        s.add_task(Task(id="c", name="C", dependencies=["b"]), noop_coro)
        await s.run()

        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.FAILED
        assert s.get_task("c").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_independent_tasks_not_affected_by_failure(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=0), failing_coro)
        s.add_task(Task(id="b", name="B"), noop_coro)  # independent
        await s.run()

        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("b").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_partial_group_failure(self) -> None:
        """If one task in a parallel group fails, unrelated branches survive."""
        s = TaskScheduler()
        s.add_task(Task(id="root", name="Root"), noop_coro)
        s.add_task(
            Task(id="fail_branch", name="Fail", dependencies=["root"], max_retries=0),
            failing_coro,
        )
        s.add_task(
            Task(id="ok_branch", name="OK", dependencies=["root"]),
            noop_coro,
        )
        s.add_task(
            Task(id="downstream_fail", name="DF", dependencies=["fail_branch"]),
            noop_coro,
        )
        s.add_task(
            Task(id="downstream_ok", name="DO", dependencies=["ok_branch"]),
            noop_coro,
        )
        await s.run()

        assert s.get_task("root").status == TaskStatus.COMPLETED
        assert s.get_task("fail_branch").status == TaskStatus.FAILED
        assert s.get_task("ok_branch").status == TaskStatus.COMPLETED
        assert s.get_task("downstream_fail").status == TaskStatus.FAILED
        assert s.get_task("downstream_ok").status == TaskStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════
# Observer pattern
# ═══════════════════════════════════════════════════════════════════


class TestObserverPattern:
    @pytest.mark.asyncio
    async def test_on_task_start_fires(self) -> None:
        started: list[str] = []
        s = TaskScheduler()
        s.on("on_task_start", lambda t: started.append(t.id))
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_fires(self) -> None:
        completed: list[str] = []
        s = TaskScheduler()
        s.on("on_task_complete", lambda t: completed.append(t.id))
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_fires(self) -> None:
        failed: list[str] = []
        s = TaskScheduler()
        s.on("on_task_fail", lambda t: failed.append(t.id))
        s.add_task(Task(id="a", name="A", max_retries=0), failing_coro)
        await s.run()
        assert "a" in failed

    @pytest.mark.asyncio
    async def test_async_listener(self) -> None:
        results: list[str] = []

        async def async_listener(task: Task) -> None:
            results.append(f"async-{task.id}")

        s = TaskScheduler()
        s.on("on_task_complete", async_listener)
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        assert "async-a" in results

    @pytest.mark.asyncio
    async def test_multiple_listeners(self) -> None:
        log1: list[str] = []
        log2: list[str] = []

        s = TaskScheduler()
        s.on("on_task_complete", lambda t: log1.append(t.id))
        s.on("on_task_complete", lambda t: log2.append(t.id))
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        assert log1 == ["a"]
        assert log2 == ["a"]


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════


class TestMetrics:
    @pytest.mark.asyncio
    async def test_total_time_measured(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), slow_coro)
        metrics = await s.run()
        assert metrics.total_time >= 0.04  # at least ~50ms

    @pytest.mark.asyncio
    async def test_per_task_time_populated(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        s.add_task(Task(id="b", name="B"), noop_coro)
        metrics = await s.run()
        assert "a" in metrics.per_task_time
        assert "b" in metrics.per_task_time

    @pytest.mark.asyncio
    async def test_retry_counts_in_metrics(self) -> None:
        coro = _make_failing_after(1)
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A", max_retries=3), coro)
        metrics = await s.run()
        assert metrics.retry_counts["a"] == 1

    @pytest.mark.asyncio
    async def test_metrics_property_returns_same_object(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        assert s.metrics is s._metrics


# ═══════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        s.reset()
        assert s.get_task("a").status == TaskStatus.PENDING
        assert s.get_task("a").result is None
        assert s.get_task("a").retry_count == 0
        assert s.metrics.total_time == 0.0

    @pytest.mark.asyncio
    async def test_can_rerun_after_reset(self) -> None:
        s = TaskScheduler()
        s.add_task(Task(id="a", name="A"), noop_coro)
        await s.run()
        s.reset()
        metrics = await s.run()

        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert metrics.total_time > 0


# ═══════════════════════════════════════════════════════════════════
# CircularDependencyError
# ═══════════════════════════════════════════════════════════════════


class TestCircularDependencyError:
    def test_message_without_cycle(self) -> None:
        err = CircularDependencyError()
        assert str(err) == "Circular dependency detected"
        assert err.cycle == []

    def test_message_with_cycle(self) -> None:
        err = CircularDependencyError(cycle=["a", "b", "a"])
        assert "a -> b -> a" in str(err)
        assert err.cycle == ["a", "b", "a"]


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_scheduler(self) -> None:
        s = TaskScheduler()
        metrics = await s.run()
        assert metrics.total_time >= 0
        assert metrics.per_task_time == {}

    @pytest.mark.asyncio
    async def test_single_task_no_deps(self) -> None:
        s = TaskScheduler(max_concurrency=1)
        s.add_task(Task(id="solo", name="Solo"), noop_coro)
        await s.run()
        assert s.get_task("solo").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_many_independent_tasks(self) -> None:
        s = TaskScheduler(max_concurrency=10)
        for i in range(20):
            s.add_task(Task(id=f"t{i}", name=f"T{i}"), noop_coro)
        await s.run()
        for i in range(20):
            assert s.get_task(f"t{i}").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_deep_chain(self) -> None:
        s = TaskScheduler()
        prev_id: str | None = None
        for i in range(10):
            tid = f"step{i}"
            deps = [prev_id] if prev_id else []
            s.add_task(Task(id=tid, name=f"Step {i}", dependencies=deps), noop_coro)
            prev_id = tid
        await s.run()
        for i in range(10):
            assert s.get_task(f"step{i}").status == TaskStatus.COMPLETED
