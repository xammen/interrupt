"""
Comprehensive pytest test suite for task_scheduler.py.

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
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(
    id: str,
    name: str = None,
    priority: int = 5,
    dependencies: list[str] = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        id=id,
        name=name or id,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


def success_fn(return_value=None):
    """Returns a zero-arg sync callable that returns *return_value*."""
    def fn():
        return return_value
    return fn


def async_success_fn(return_value=None):
    """Returns a zero-arg async callable that returns *return_value*."""
    async def fn():
        return return_value
    return fn


def failing_fn(exc=None):
    """Returns a zero-arg callable that always raises."""
    if exc is None:
        exc = RuntimeError("deliberate failure")
    def fn():
        raise exc
    return fn


def async_failing_fn(exc=None):
    """Returns a zero-arg async callable that always raises."""
    if exc is None:
        exc = RuntimeError("deliberate async failure")
    async def fn():
        raise exc
    return fn


def flaky_fn(fail_times: int, return_value=None):
    """Raises on the first *fail_times* calls, then succeeds."""
    calls = {"count": 0}
    def fn():
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise RuntimeError(f"flaky failure #{calls['count']}")
        return return_value
    return fn


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_priority_boundaries(self):
        t1 = make_task("a", priority=1)
        t10 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_zero_raises(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=0)

    def test_invalid_priority_eleven_raises(self):
        with pytest.raises(ValueError, match="priority"):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("a")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_zero(self):
        t = make_task("a")
        assert t.retry_count == 0

    def test_default_result_is_none(self):
        t = make_task("a")
        assert t.result is None

    def test_dependencies_default_empty(self):
        t = make_task("a")
        assert t.dependencies == []


# ---------------------------------------------------------------------------
# TaskMetrics tests
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_calculation(self):
        m = TaskMetrics(task_id="a", start_time=10.0, end_time=13.5)
        assert m.elapsed == pytest.approx(3.5)

    def test_elapsed_zero_when_not_run(self):
        m = TaskMetrics(task_id="a")
        assert m.elapsed == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TaskScheduler initialisation
# ---------------------------------------------------------------------------

class TestTaskSchedulerInit:
    def test_default_concurrency(self):
        s = TaskScheduler()
        assert s.max_concurrency == 4

    def test_custom_concurrency(self):
        s = TaskScheduler(max_concurrency=2)
        assert s.max_concurrency == 2

    def test_zero_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            TaskScheduler(max_concurrency=0)

    def test_negative_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            TaskScheduler(max_concurrency=-1)


# ---------------------------------------------------------------------------
# add_task / registration tests
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_single_task(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        assert "a" in s._tasks

    def test_duplicate_task_id_raises(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        with pytest.raises(ValueError, match="already registered"):
            s.add_task(make_task("a"), success_fn())

    def test_metrics_initialised_on_add(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        assert "a" in s._metrics
        assert isinstance(s._metrics["a"], TaskMetrics)


# ---------------------------------------------------------------------------
# Observer / event subscription
# ---------------------------------------------------------------------------

class TestObserver:
    def test_on_registers_callback(self):
        s = TaskScheduler()
        cb = MagicMock()
        s.on("on_task_start", cb)
        assert cb in s._listeners["on_task_start"]

    def test_unknown_event_raises(self):
        s = TaskScheduler()
        with pytest.raises(ValueError, match="Unknown event"):
            s.on("on_task_banana", lambda t: None)

    @pytest.mark.asyncio
    async def test_sync_callback_called_on_start(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn("result"))
        started = []
        s.on("on_task_start", lambda task: started.append(task.id))
        await s.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_sync_callback_called_on_complete(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn("result"))
        completed = []
        s.on("on_task_complete", lambda task: completed.append(task.id))
        await s.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_async_callback_called_on_start(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        called = []
        async def async_cb(task):
            called.append(task.id)
        s.on("on_task_start", async_cb)
        await s.run()
        assert "a" in called

    @pytest.mark.asyncio
    async def test_fail_callback_called_on_failure(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=0)
        s.add_task(t, failing_fn())
        failed = []
        s.on("on_task_fail", lambda task: failed.append(task.id))
        with pytest.raises(Exception):
            await s.run()
        assert "a" in failed

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        calls = []
        s.on("on_task_complete", lambda task: calls.append(1))
        s.on("on_task_complete", lambda task: calls.append(2))
        await s.run()
        assert calls == [1, 2]


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_sync_task_completes(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn(42))
        await s.run()
        assert s._tasks["a"].status == TaskStatus.COMPLETED
        assert s._tasks["a"].result == 42

    @pytest.mark.asyncio
    async def test_single_async_task_completes(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, async_success_fn("hello"))
        await s.run()
        assert s._tasks["a"].status == TaskStatus.COMPLETED
        assert s._tasks["a"].result == "hello"

    @pytest.mark.asyncio
    async def test_run_returns_metrics_dict(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        metrics = await s.run()
        assert isinstance(metrics, dict)
        assert "a" in metrics

    @pytest.mark.asyncio
    async def test_metrics_record_timing(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, success_fn())
        await s.run()
        m = s._metrics["a"]
        assert m.start_time > 0
        assert m.end_time >= m.start_time

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        s = TaskScheduler()
        for i in range(5):
            s.add_task(make_task(f"t{i}"), success_fn(i))
        await s.run()
        for i in range(5):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED
            assert s._tasks[f"t{i}"].result == i

    @pytest.mark.asyncio
    async def test_total_elapsed_populated_after_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn())
        await s.run()
        assert s.total_elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_returns_snapshot(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn())
        await s.run()
        snap = s.get_metrics()
        assert "a" in snap
        assert isinstance(snap["a"], TaskMetrics)

    @pytest.mark.asyncio
    async def test_empty_scheduler_runs_without_error(self):
        s = TaskScheduler()
        metrics = await s.run()
        assert metrics == {}


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    @pytest.mark.asyncio
    async def test_dependent_task_runs_after_dependency(self):
        order = []
        s = TaskScheduler()
        s.add_task(make_task("a"), lambda: order.append("a"))
        s.add_task(make_task("b", dependencies=["a"]), lambda: order.append("b"))
        await s.run()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_chain_a_b_c_executes_in_order(self):
        order = []
        s = TaskScheduler()
        s.add_task(make_task("a"), lambda: order.append("a"))
        s.add_task(make_task("b", dependencies=["a"]), lambda: order.append("b"))
        s.add_task(make_task("c", dependencies=["b"]), lambda: order.append("c"))
        await s.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """a -> b, a -> c, b+c -> d"""
        order = []
        s = TaskScheduler()
        s.add_task(make_task("a"), lambda: order.append("a"))
        s.add_task(make_task("b", dependencies=["a"]), lambda: order.append("b"))
        s.add_task(make_task("c", dependencies=["a"]), lambda: order.append("c"))
        s.add_task(make_task("d", dependencies=["b", "c"]), lambda: order.append("d"))
        await s.run()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    @pytest.mark.asyncio
    async def test_dependency_result_available_when_dependent_runs(self):
        results = {}
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn(99))

        def read_result():
            results["saw_a_result"] = s._tasks["a"].result
        s.add_task(make_task("b", dependencies=["a"]), read_result)
        await s.run()
        assert results["saw_a_result"] == 99

    def test_unknown_dependency_raises_on_get_plan(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["ghost"]), success_fn())
        with pytest.raises(ValueError, match="unknown task"):
            s.get_execution_plan()

    @pytest.mark.asyncio
    async def test_unknown_dependency_raises_on_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["ghost"]), success_fn())
        with pytest.raises(ValueError, match="unknown task"):
            await s.run()

    def test_get_execution_plan_returns_groups(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn())
        s.add_task(make_task("b", dependencies=["a"]), success_fn())
        plan = s.get_execution_plan()
        assert isinstance(plan, list)
        assert len(plan) >= 2
        # "a" must appear in an earlier group than "b"
        group_of = {}
        for i, group in enumerate(plan):
            for tid in group:
                group_of[tid] = i
        assert group_of["a"] < group_of["b"]

    def test_get_execution_plan_no_deps_single_group(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn())
        s.add_task(make_task("b"), success_fn())
        plan = s.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_priority_ordering_within_group(self):
        """Higher priority tasks should appear first within a group."""
        s = TaskScheduler()
        s.add_task(make_task("lo", priority=1), success_fn())
        s.add_task(make_task("hi", priority=9), success_fn())
        plan = s.get_execution_plan()
        group = plan[0]
        assert group.index("hi") < group.index("lo")


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------

class TestCircularDependencyDetection:
    def test_self_loop_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_two_node_cycle_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), success_fn())
        s.add_task(make_task("b", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_three_node_cycle_raises(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["c"]), success_fn())
        s.add_task(make_task("b", dependencies=["a"]), success_fn())
        s.add_task(make_task("c", dependencies=["b"]), success_fn())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_cycle_error_contains_cycle_attribute(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), success_fn())
        s.add_task(make_task("b", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert isinstance(exc_info.value.cycle, list)
        assert len(exc_info.value.cycle) >= 2

    def test_cycle_error_message_contains_arrow(self):
        s = TaskScheduler()
        s.add_task(make_task("x", dependencies=["y"]), success_fn())
        s.add_task(make_task("y", dependencies=["x"]), success_fn())
        with pytest.raises(CircularDependencyError) as exc_info:
            s.get_execution_plan()
        assert "->" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circular_dependency_raises_on_run(self):
        s = TaskScheduler()
        s.add_task(make_task("a", dependencies=["b"]), success_fn())
        s.add_task(make_task("b", dependencies=["a"]), success_fn())
        with pytest.raises(CircularDependencyError):
            await s.run()

    def test_partial_cycle_in_larger_graph(self):
        """Graph has valid tasks and a cyclic sub-graph."""
        s = TaskScheduler()
        s.add_task(make_task("ok1"), success_fn())
        s.add_task(make_task("ok2", dependencies=["ok1"]), success_fn())
        s.add_task(make_task("bad1", dependencies=["bad2"]), success_fn())
        s.add_task(make_task("bad2", dependencies=["bad1"]), success_fn())
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_and_eventually_succeeds(self):
        """Task fails twice then succeeds on the third attempt."""
        s = TaskScheduler()
        t = make_task("a", max_retries=3)
        s.add_task(t, flaky_fn(fail_times=2, return_value="ok"))
        await s.run()
        assert s._tasks["a"].status == TaskStatus.COMPLETED
        assert s._tasks["a"].result == "ok"
        assert s._tasks["a"].retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_exhausting_retries(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=2)
        s.add_task(t, failing_fn(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            await s.run()
        assert s._tasks["a"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_count_increments_correctly(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=3)
        s.add_task(t, failing_fn())
        with pytest.raises(Exception):
            await s.run()
        assert s._tasks["a"].retry_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=0)
        s.add_task(t, failing_fn(ValueError("no retry")))
        with pytest.raises(ValueError, match="no retry"):
            await s.run()
        assert s._tasks["a"].retry_count == 0

    @pytest.mark.asyncio
    async def test_backoff_delays_are_exponential(self):
        """Verify asyncio.sleep is called with 2**0, 2**1 for 2 retries."""
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        s = TaskScheduler()
        t = make_task("a", max_retries=2)
        s.add_task(t, failing_fn())

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(Exception):
                await s.run()

        assert sleep_calls == [1, 2]  # 2**0=1, 2**1=2

    @pytest.mark.asyncio
    async def test_backoff_three_retries(self):
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        s = TaskScheduler()
        t = make_task("a", max_retries=3)
        s.add_task(t, failing_fn())

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(Exception):
                await s.run()

        assert sleep_calls == [1, 2, 4]  # 2**0, 2**1, 2**2

    @pytest.mark.asyncio
    async def test_metrics_retries_field_set_on_failure(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=2)
        s.add_task(t, failing_fn())

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception):
                await s.run()

        assert s._metrics["a"].retries == 2

    @pytest.mark.asyncio
    async def test_metrics_retries_field_set_on_success_after_retry(self):
        s = TaskScheduler()
        t = make_task("a", max_retries=3)
        s.add_task(t, flaky_fn(fail_times=1))

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            await s.run()

        assert s._metrics["a"].retries == 1

    @pytest.mark.asyncio
    async def test_no_sleep_on_first_attempt_success(self):
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn())

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            await s.run()

        assert sleep_calls == []


# ---------------------------------------------------------------------------
# Concurrent execution respecting concurrency limits
# ---------------------------------------------------------------------------

class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_concurrency_limit_one_sequential(self):
        """With max_concurrency=1 tasks must not overlap."""
        active = {"count": 0, "max": 0}

        async def tracked():
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0)  # yield to event loop
            active["count"] -= 1

        s = TaskScheduler(max_concurrency=1)
        for i in range(4):
            s.add_task(make_task(f"t{i}"), tracked)

        await s.run()
        assert active["max"] == 1

    @pytest.mark.asyncio
    async def test_concurrency_limit_two(self):
        """With max_concurrency=2 at most 2 tasks overlap."""
        active = {"count": 0, "max": 0}

        async def tracked():
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.01)
            active["count"] -= 1

        s = TaskScheduler(max_concurrency=2)
        for i in range(6):
            s.add_task(make_task(f"t{i}"), tracked)

        await s.run()
        assert active["max"] <= 2

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently(self):
        """Without a concurrency limit of 1, independent tasks should overlap."""
        started = []
        finished = []
        barrier = asyncio.Event()

        async def wait_task(tid):
            started.append(tid)
            await barrier.wait()
            finished.append(tid)

        s = TaskScheduler(max_concurrency=4)
        for i in range(4):
            tid = f"t{i}"
            s.add_task(make_task(tid), lambda t=tid: wait_task(t))

        async def runner():
            run_coro = asyncio.create_task(s.run())
            # Give the event loop a moment for all tasks to start
            for _ in range(10):
                await asyncio.sleep(0)
            assert len(started) == 4, "All 4 tasks should have started concurrently"
            barrier.set()
            await run_coro

        await runner()

    @pytest.mark.asyncio
    async def test_semaphore_respected_across_groups(self):
        """Even when tasks span multiple dependency groups the semaphore is respected."""
        active = {"count": 0, "max": 0}

        async def tracked():
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.01)
            active["count"] -= 1

        s = TaskScheduler(max_concurrency=2)
        # Group 1: a, b
        s.add_task(make_task("a"), tracked)
        s.add_task(make_task("b"), tracked)
        # Group 2: c, d (both depend on a and b)
        s.add_task(make_task("c", dependencies=["a", "b"]), tracked)
        s.add_task(make_task("d", dependencies=["a", "b"]), tracked)

        await s.run()
        assert active["max"] <= 2

    @pytest.mark.asyncio
    async def test_all_tasks_complete_despite_concurrency_limit(self):
        s = TaskScheduler(max_concurrency=2)
        for i in range(8):
            s.add_task(make_task(f"t{i}"), success_fn(i))
        await s.run()
        for i in range(8):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_high_concurrency_does_not_break_execution(self):
        """Setting concurrency higher than task count is fine."""
        s = TaskScheduler(max_concurrency=100)
        for i in range(5):
            s.add_task(make_task(f"t{i}"), success_fn(i))
        await s.run()
        for i in range(5):
            assert s._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_single_concurrency_with_dependencies_preserves_order(self):
        order = []
        s = TaskScheduler(max_concurrency=1)
        s.add_task(make_task("a"), lambda: order.append("a"))
        s.add_task(make_task("b", dependencies=["a"]), lambda: order.append("b"))
        s.add_task(make_task("c", dependencies=["b"]), lambda: order.append("c"))
        await s.run()
        assert order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Edge-case / integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_fan_out_fan_in(self):
        """One root feeds N workers which all feed one sink."""
        order = []
        N = 5
        s = TaskScheduler(max_concurrency=N)
        s.add_task(make_task("root"), lambda: order.append("root"))
        for i in range(N):
            wid = f"w{i}"
            s.add_task(make_task(wid, dependencies=["root"]), lambda w=wid: order.append(w))
        s.add_task(
            make_task("sink", dependencies=[f"w{i}" for i in range(N)]),
            lambda: order.append("sink"),
        )
        await s.run()
        assert order[0] == "root"
        assert order[-1] == "sink"
        assert set(order[1:-1]) == {f"w{i}" for i in range(N)}

    @pytest.mark.asyncio
    async def test_failed_task_status_is_failed(self):
        s = TaskScheduler()
        t = make_task("bad", max_retries=0)
        s.add_task(t, failing_fn())
        with pytest.raises(Exception):
            await s.run()
        assert s._tasks["bad"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_multiple_failures_in_group_raises_runtime_error(self):
        s = TaskScheduler(max_concurrency=4)
        for i in range(3):
            s.add_task(make_task(f"bad{i}", max_retries=0), failing_fn())

        with patch("task_scheduler.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises((RuntimeError, Exception)):
                await s.run()

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_tasks(self):
        s = TaskScheduler()
        s.add_task(make_task("sync"), success_fn("sync_result"))
        s.add_task(make_task("async"), async_success_fn("async_result"))
        await s.run()
        assert s._tasks["sync"].result == "sync_result"
        assert s._tasks["async"].result == "async_result"

    @pytest.mark.asyncio
    async def test_task_result_none_is_valid(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn(None))
        await s.run()
        assert s._tasks["a"].status == TaskStatus.COMPLETED
        assert s._tasks["a"].result is None

    @pytest.mark.asyncio
    async def test_complex_result_stored(self):
        payload = {"key": [1, 2, 3], "nested": {"x": True}}
        s = TaskScheduler()
        s.add_task(make_task("a"), success_fn(payload))
        await s.run()
        assert s._tasks["a"].result == payload
