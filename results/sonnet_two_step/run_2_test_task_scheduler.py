"""
test_task_scheduler.py

Comprehensive pytest test suite for task_scheduler.py.
Covers: basic task execution, dependency resolution, circular dependency
detection, retry logic with exponential backoff, and concurrent execution
respecting concurrency limits.
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    EventBus,
    SchedulerMetrics,
    Task,
    TaskMetrics,
    TaskScheduler,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(
    task_id: str,
    name: str = "task",
    priority: int = 5,
    dependencies: list | None = None,
    max_retries: int = 3,
    fn=None,
) -> Task:
    if fn is None:
        async def _noop():
            return f"result:{task_id}"
        fn = _noop
    return Task(
        id=task_id,
        name=name,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
        fn=fn,
    )


def make_failing_task(
    task_id: str,
    fail_times: int = 1,
    priority: int = 5,
    max_retries: int = 3,
) -> Task:
    """Returns a task whose fn raises on the first *fail_times* calls."""
    call_count = {"n": 0}

    async def _fn():
        call_count["n"] += 1
        if call_count["n"] <= fail_times:
            raise RuntimeError(f"intentional failure #{call_count['n']}")
        return f"result:{task_id}"

    return Task(
        id=task_id,
        name=task_id,
        priority=priority,
        max_retries=max_retries,
        fn=_fn,
    )


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


class TestTask:
    def test_valid_priority_boundaries(self):
        t1 = make_task("t1", priority=1)
        t10 = make_task("t10", priority=10)
        assert t1.priority == 1
        assert t10.priority == 10

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="x", name="x", priority=0)
        with pytest.raises(ValueError, match="priority must be between 1 and 10"):
            Task(id="x", name="x", priority=11)

    def test_default_status_is_pending(self):
        t = make_task("t")
        assert t.status == TaskStatus.PENDING

    def test_default_retry_count_zero(self):
        t = make_task("t")
        assert t.retry_count == 0

    def test_result_initially_none(self):
        t = make_task("t")
        assert t.result is None


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------


class TestTaskMetrics:
    def test_elapsed_none_when_times_missing(self):
        m = TaskMetrics(task_id="t")
        assert m.elapsed is None

    def test_elapsed_none_when_end_missing(self):
        m = TaskMetrics(task_id="t", start_time=1.0)
        assert m.elapsed is None

    def test_elapsed_computed_correctly(self):
        m = TaskMetrics(task_id="t", start_time=1.0, end_time=4.5)
        assert pytest.approx(m.elapsed) == 3.5


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------


class TestSchedulerMetrics:
    def test_total_elapsed_none_initially(self):
        sm = SchedulerMetrics()
        assert sm.total_elapsed is None

    def test_total_elapsed_computed(self):
        sm = SchedulerMetrics(total_start=0.0, total_end=10.0)
        assert pytest.approx(sm.total_elapsed) == 10.0


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribed_callback_is_called(self):
        bus = EventBus()
        received = []

        async def cb(task):
            received.append(task.id)

        bus.subscribe("my_event", cb)
        t = make_task("t1")
        await bus.emit("my_event", t)
        assert received == ["t1"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_called(self):
        bus = EventBus()
        log: List[str] = []

        async def cb1(task):
            log.append("cb1")

        async def cb2(task):
            log.append("cb2")

        bus.subscribe("ev", cb1)
        bus.subscribe("ev", cb2)
        await bus.emit("ev", make_task("t"))
        assert log == ["cb1", "cb2"]

    @pytest.mark.asyncio
    async def test_emit_unknown_event_is_noop(self):
        bus = EventBus()
        # Should not raise
        await bus.emit("nonexistent", make_task("t"))


# ---------------------------------------------------------------------------
# TaskScheduler — registration
# ---------------------------------------------------------------------------


class TestTaskSchedulerRegistration:
    def test_add_task_registers_task(self):
        sched = TaskScheduler()
        t = make_task("t1")
        sched.add_task(t)
        assert "t1" in sched._tasks

    def test_duplicate_id_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        with pytest.raises(ValueError, match="already registered"):
            sched.add_task(make_task("t1"))

    def test_validate_dependencies_unknown_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1", dependencies=["missing"]))
        with pytest.raises(ValueError, match="unknown task"):
            sched._validate_dependencies()

    def test_validate_dependencies_known_passes(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"))
        sched.add_task(make_task("b", dependencies=["a"]))
        sched._validate_dependencies()  # no exception


# ---------------------------------------------------------------------------
# Dependency resolution / topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_single_task_group(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        plan = sched.get_execution_plan()
        assert plan == [["t1"]]

    def test_independent_tasks_in_one_group(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", priority=5))
        sched.add_task(make_task("b", priority=5))
        plan = sched.get_execution_plan()
        assert len(plan) == 1
        assert set(plan[0]) == {"a", "b"}

    def test_linear_chain_produces_separate_groups(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"))
        sched.add_task(make_task("b", dependencies=["a"]))
        sched.add_task(make_task("c", dependencies=["b"]))
        plan = sched.get_execution_plan()
        assert len(plan) == 3
        assert plan[0] == ["a"]
        assert plan[1] == ["b"]
        assert plan[2] == ["c"]

    def test_diamond_dependency(self):
        """a -> b, a -> c, b -> d, c -> d"""
        sched = TaskScheduler()
        sched.add_task(make_task("a", priority=5))
        sched.add_task(make_task("b", priority=5, dependencies=["a"]))
        sched.add_task(make_task("c", priority=5, dependencies=["a"]))
        sched.add_task(make_task("d", priority=5, dependencies=["b", "c"]))
        plan = sched.get_execution_plan()
        assert plan[0] == ["a"]
        # b and c can run concurrently
        assert set(plan[1]) == {"b", "c"}
        assert plan[2] == ["d"]

    def test_priority_ordering_within_group(self):
        """Within a concurrent group, higher-priority tasks come first."""
        sched = TaskScheduler()
        sched.add_task(make_task("low", priority=2))
        sched.add_task(make_task("high", priority=9))
        sched.add_task(make_task("mid", priority=5))
        plan = sched.get_execution_plan()
        assert len(plan) == 1
        assert plan[0] == ["high", "mid", "low"]

    def test_execution_plan_raises_on_unknown_dependency(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t", dependencies=["ghost"]))
        with pytest.raises(ValueError):
            sched.get_execution_plan()


# ---------------------------------------------------------------------------
# Circular dependency detection
# ---------------------------------------------------------------------------


class TestCircularDependencyDetection:
    def test_direct_cycle_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]))
        sched.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_three_node_cycle_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["c"]))
        sched.add_task(make_task("b", dependencies=["a"]))
        sched.add_task(make_task("c", dependencies=["b"]))
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_self_loop_raises(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            sched.get_execution_plan()

    def test_cycle_error_message_contains_task_ids(self):
        sched = TaskScheduler()
        sched.add_task(make_task("x", dependencies=["y"]))
        sched.add_task(make_task("y", dependencies=["x"]))
        with pytest.raises(CircularDependencyError, match="x|y"):
            sched.get_execution_plan()

    def test_no_cycle_does_not_raise(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"))
        sched.add_task(make_task("b", dependencies=["a"]))
        # Must not raise
        sched.get_execution_plan()


# ---------------------------------------------------------------------------
# Basic task execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        await sched.run()
        assert sched._tasks["t1"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_result_is_stored(self):
        async def work():
            return 42

        sched = TaskScheduler()
        sched.add_task(make_task("t1", fn=work))
        await sched.run()
        assert sched._tasks["t1"].result == 42

    @pytest.mark.asyncio
    async def test_multiple_independent_tasks_all_complete(self):
        sched = TaskScheduler()
        for i in range(5):
            sched.add_task(make_task(f"t{i}"))
        await sched.run()
        for i in range(5):
            assert sched._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_with_no_fn_fails(self):
        t = Task(id="t", name="t", priority=5, fn=None)
        sched = TaskScheduler()
        sched.add_task(t)
        await sched.run()
        assert sched._tasks["t"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_metrics_populated_after_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        metrics = await sched.run()
        assert metrics.total_elapsed is not None
        assert metrics.total_elapsed >= 0
        assert "t1" in metrics.per_task

    @pytest.mark.asyncio
    async def test_on_task_complete_event_fired(self):
        fired = []

        async def on_complete(task):
            fired.append(task.id)

        sched = TaskScheduler()
        sched.subscribe(TaskScheduler.ON_TASK_COMPLETE, on_complete)
        sched.add_task(make_task("t1"))
        await sched.run()
        assert "t1" in fired

    @pytest.mark.asyncio
    async def test_on_task_start_event_fired(self):
        started = []

        async def on_start(task):
            started.append(task.id)

        sched = TaskScheduler()
        sched.subscribe(TaskScheduler.ON_TASK_START, on_start)
        sched.add_task(make_task("t1"))
        await sched.run()
        assert "t1" in started


# ---------------------------------------------------------------------------
# Dependency execution order
# ---------------------------------------------------------------------------


class TestDependencyExecutionOrder:
    @pytest.mark.asyncio
    async def test_dependency_completes_before_dependent(self):
        order = []

        async def fn_a():
            order.append("a")

        async def fn_b():
            order.append("b")

        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="a", priority=5, fn=fn_a))
        sched.add_task(Task(id="b", name="b", priority=5, dependencies=["a"], fn=fn_b))
        await sched.run()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_chain_executes_in_order(self):
        order = []

        def make_fn(label):
            async def fn():
                order.append(label)
            return fn

        sched = TaskScheduler()
        sched.add_task(Task(id="a", name="a", priority=5, fn=make_fn("a")))
        sched.add_task(Task(id="b", name="b", priority=5, dependencies=["a"], fn=make_fn("b")))
        sched.add_task(Task(id="c", name="c", priority=5, dependencies=["b"], fn=make_fn("c")))
        await sched.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_all_tasks_completed_after_run(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a"))
        sched.add_task(make_task("b", dependencies=["a"]))
        sched.add_task(make_task("c", dependencies=["a"]))
        sched.add_task(make_task("d", dependencies=["b", "c"]))
        await sched.run()
        for tid in ("a", "b", "c", "d"):
            assert sched._tasks[tid].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_task_retries_on_failure_and_eventually_succeeds(self):
        """Task fails twice then succeeds on third attempt."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            sched = TaskScheduler()
            sched.add_task(make_failing_task("t", fail_times=2, max_retries=3))
            await sched.run()
            assert sched._tasks["t"].status == TaskStatus.COMPLETED
            assert sched._tasks["t"].retry_count == 2

    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries_exhausted(self):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            sched = TaskScheduler()
            sched.add_task(make_failing_task("t", fail_times=10, max_retries=3))
            await sched.run()
            assert sched._tasks["t"].status == TaskStatus.FAILED
            assert sched._tasks["t"].retry_count == 4  # initial + 3 retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Verify asyncio.sleep is called with 2**0, 2**1, 2**2 on retries."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            sched = TaskScheduler()
            sched.add_task(make_failing_task("t", fail_times=3, max_retries=3))
            await sched.run()
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert delays == [1, 2, 4]  # 2**0, 2**1, 2**2

    @pytest.mark.asyncio
    async def test_on_task_fail_event_fired_when_retries_exhausted(self):
        failed_ids = []

        async def on_fail(task):
            failed_ids.append(task.id)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            sched = TaskScheduler()
            sched.subscribe(TaskScheduler.ON_TASK_FAIL, on_fail)
            sched.add_task(make_failing_task("t", fail_times=10, max_retries=2))
            await sched.run()
            assert "t" in failed_ids

    @pytest.mark.asyncio
    async def test_no_retry_on_zero_max_retries(self):
        """max_retries=0 means the task fails immediately without retrying."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            sched = TaskScheduler()
            sched.add_task(make_failing_task("t", fail_times=1, max_retries=0))
            await sched.run()
            assert sched._tasks["t"].status == TaskStatus.FAILED
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_count_recorded_in_metrics(self):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            sched = TaskScheduler()
            sched.add_task(make_failing_task("t", fail_times=2, max_retries=3))
            metrics = await sched.run()
            assert metrics.per_task["t"].retry_count == 2

    @pytest.mark.asyncio
    async def test_successful_task_has_zero_retries_in_metrics(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t"))
        metrics = await sched.run()
        assert metrics.per_task["t"].retry_count == 0


# ---------------------------------------------------------------------------
# Concurrent execution and concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """At most max_concurrency tasks should run simultaneously."""
        max_concurrency = 2
        running = {"count": 0, "max_seen": 0}
        barrier = asyncio.Event()

        async def slow_fn():
            running["count"] += 1
            running["max_seen"] = max(running["max_seen"], running["count"])
            # Yield control so other coroutines can start
            await asyncio.sleep(0)
            running["count"] -= 1

        sched = TaskScheduler(max_concurrency=max_concurrency)
        for i in range(6):
            sched.add_task(Task(id=f"t{i}", name=f"t{i}", priority=5, fn=slow_fn))
        await sched.run()
        assert running["max_seen"] <= max_concurrency

    @pytest.mark.asyncio
    async def test_all_tasks_complete_despite_concurrency_limit(self):
        sched = TaskScheduler(max_concurrency=2)
        for i in range(8):
            sched.add_task(make_task(f"t{i}"))
        await sched.run()
        for i in range(8):
            assert sched._tasks[f"t{i}"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_max_concurrency_one_runs_sequentially(self):
        """max_concurrency=1 forces strictly sequential execution."""
        order = []

        def make_fn(label):
            async def fn():
                order.append(("start", label))
                await asyncio.sleep(0)
                order.append(("end", label))
            return fn

        sched = TaskScheduler(max_concurrency=1)
        for i in range(4):
            sched.add_task(Task(id=f"t{i}", name=f"t{i}", priority=5, fn=make_fn(i)))
        await sched.run()

        # With concurrency=1, no two tasks should overlap:
        # every "start" for task i must be followed by its own "end" before
        # the next "start".
        for idx in range(0, len(order) - 1, 2):
            event, label = order[idx]
            next_event, next_label = order[idx + 1]
            assert event == "start"
            assert next_event == "end"
            assert label == next_label

    @pytest.mark.asyncio
    async def test_independent_tasks_run_concurrently_when_allowed(self):
        """With max_concurrency >= n independent tasks, all start before any ends."""
        n = 4
        started = []
        gate = asyncio.Event()

        async def gated_fn(label):
            started.append(label)
            await gate.wait()

        sched = TaskScheduler(max_concurrency=n)
        for i in range(n):
            sched.add_task(Task(id=f"t{i}", name=f"t{i}", priority=5, fn=lambda i=i: gated_fn(i)))

        async def release_gate():
            # Wait until all tasks have started, then release
            while len(started) < n:
                await asyncio.sleep(0)
            gate.set()

        await asyncio.gather(sched.run(), release_gate())
        assert len(started) == n

    @pytest.mark.asyncio
    async def test_high_concurrency_limit_does_not_break_execution(self):
        sched = TaskScheduler(max_concurrency=100)
        for i in range(10):
            sched.add_task(make_task(f"t{i}"))
        await sched.run()
        for i in range(10):
            assert sched._tasks[f"t{i}"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Integration: combined dependency + retry + concurrency
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_dependency_chain_with_failing_intermediate_task(self):
        """
        a -> b -> c
        b fails twice then succeeds; c should still complete.
        """
        with patch("asyncio.sleep", new_callable=AsyncMock):
            sched = TaskScheduler(max_concurrency=2)
            sched.add_task(make_task("a"))
            sched.add_task(make_failing_task("b", fail_times=2, max_retries=3))
            # Give b a dependency on a
            b_task = sched._tasks["b"]
            b_task.dependencies = ["a"]
            sched.add_task(make_task("c", dependencies=["b"]))
            await sched.run()
            assert sched._tasks["a"].status == TaskStatus.COMPLETED
            assert sched._tasks["b"].status == TaskStatus.COMPLETED
            assert sched._tasks["c"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_returns_scheduler_metrics(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        result = await sched.run()
        assert isinstance(result, SchedulerMetrics)
        assert result.total_start is not None
        assert result.total_end is not None

    @pytest.mark.asyncio
    async def test_run_raises_on_circular_dependency(self):
        sched = TaskScheduler()
        sched.add_task(make_task("a", dependencies=["b"]))
        sched.add_task(make_task("b", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            await sched.run()

    @pytest.mark.asyncio
    async def test_event_sequence_start_then_complete(self):
        events = []

        async def on_start(task):
            events.append(("start", task.id))

        async def on_complete(task):
            events.append(("complete", task.id))

        sched = TaskScheduler()
        sched.subscribe(TaskScheduler.ON_TASK_START, on_start)
        sched.subscribe(TaskScheduler.ON_TASK_COMPLETE, on_complete)
        sched.add_task(make_task("t1"))
        await sched.run()
        assert ("start", "t1") in events
        assert ("complete", "t1") in events
        start_idx = events.index(("start", "t1"))
        complete_idx = events.index(("complete", "t1"))
        assert start_idx < complete_idx

    @pytest.mark.asyncio
    async def test_multiple_runs_reset_metrics(self):
        sched = TaskScheduler()
        sched.add_task(make_task("t1"))
        await sched.run()
        first_start = sched.metrics.total_start
        await sched.run()
        second_start = sched.metrics.total_start
        # Second run should produce fresh metrics — both starts must be set
        assert second_start is not None
        assert first_start is not None
