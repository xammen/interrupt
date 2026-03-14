"""
test_task_scheduler.py - Comprehensive tests for task_scheduler.py.

Covers:
- Task dataclass validation
- TaskMetrics / SchedulerMetrics properties
- EventBus subscribe / emit
- TaskScheduler.add_task (happy path + error cases)
- get_execution_plan / topological sort + cycle detection
- run(): concurrency, ordering, dependency enforcement
- Retry logic with exponential backoff (mocked sleep)
- Observer events (on_task_start, on_task_complete, on_task_fail)
- Introspection helpers (get_task, get_all_tasks, reset)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from task_scheduler import (
    CircularDependencyError,
    EventBus,
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

async def noop(task: Task) -> str:
    """A handler that succeeds immediately."""
    return f"done:{task.id}"


async def failing(task: Task) -> None:
    """A handler that always raises."""
    raise ValueError("deliberate failure")


def make_task(
    tid: str,
    name: str = "",
    priority: int = 5,
    dependencies: list[str] | None = None,
    max_retries: int = 0,
) -> Task:
    return Task(
        id=tid,
        name=name or tid,
        priority=priority,
        dependencies=dependencies or [],
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_valid_task_creation(self):
        t = make_task("t1", priority=7)
        assert t.id == "t1"
        assert t.priority == 7
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.result is None

    def test_priority_lower_bound(self):
        with pytest.raises(ValueError, match="Priority must be between"):
            make_task("x", priority=0)

    def test_priority_upper_bound(self):
        with pytest.raises(ValueError, match="Priority must be between"):
            make_task("x", priority=11)

    def test_priority_boundary_values(self):
        t1 = make_task("a", priority=1)
        t2 = make_task("b", priority=10)
        assert t1.priority == 1
        assert t2.priority == 10

    def test_default_dependencies_are_independent_lists(self):
        t1 = make_task("a")
        t2 = make_task("b")
        t1.dependencies.append("x")
        assert t2.dependencies == []


# ---------------------------------------------------------------------------
# TaskMetrics
# ---------------------------------------------------------------------------

class TestTaskMetrics:
    def test_elapsed_none_when_not_started(self):
        tm = TaskMetrics(task_id="x")
        assert tm.elapsed is None

    def test_elapsed_none_when_not_finished(self):
        tm = TaskMetrics(task_id="x", start_time=1.0)
        assert tm.elapsed is None

    def test_elapsed_computed(self):
        tm = TaskMetrics(task_id="x", start_time=1.0, end_time=3.5)
        assert tm.elapsed == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SchedulerMetrics
# ---------------------------------------------------------------------------

class TestSchedulerMetrics:
    def test_total_time_none_before_end(self):
        sm = SchedulerMetrics(total_start=0.0)
        assert sm.total_time is None

    def test_total_time_computed(self):
        sm = SchedulerMetrics(total_start=10.0, total_end=15.0)
        assert sm.total_time == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        bus = EventBus()
        received: list[Task] = []

        async def listener(task: Task) -> None:
            received.append(task)

        bus.subscribe("on_task_complete", listener)
        t = make_task("t1")
        await bus.emit("on_task_complete", t)
        assert received == [t]

    @pytest.mark.asyncio
    async def test_multiple_listeners_same_event(self):
        bus = EventBus()
        calls: list[str] = []

        async def l1(task: Task) -> None:
            calls.append("l1")

        async def l2(task: Task) -> None:
            calls.append("l2")

        bus.subscribe("ev", l1)
        bus.subscribe("ev", l2)
        await bus.emit("ev", make_task("x"))
        assert calls == ["l1", "l2"]

    @pytest.mark.asyncio
    async def test_emit_unknown_event_is_noop(self):
        bus = EventBus()
        # Should not raise
        await bus.emit("nonexistent", make_task("x"))

    @pytest.mark.asyncio
    async def test_listener_not_called_for_other_event(self):
        bus = EventBus()
        called = []

        async def listener(task: Task) -> None:
            called.append(True)

        bus.subscribe("on_start", listener)
        await bus.emit("on_complete", make_task("x"))
        assert called == []


# ---------------------------------------------------------------------------
# TaskScheduler.add_task
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_task_registers_task(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, noop)
        assert s.get_task("a") is t

    def test_add_task_duplicate_raises(self):
        s = TaskScheduler()
        t = make_task("a")
        s.add_task(t, noop)
        with pytest.raises(ValueError, match="already registered"):
            s.add_task(make_task("a"), noop)

    def test_add_task_unknown_dependency_raises(self):
        s = TaskScheduler()
        t = make_task("b", dependencies=["a"])
        with pytest.raises(TaskNotFoundError, match="'a' not found"):
            s.add_task(t, noop)

    def test_add_task_known_dependency_succeeds(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        assert len(s.get_all_tasks()) == 2

    def test_metrics_entry_created(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        assert "a" in s.metrics.per_task


# ---------------------------------------------------------------------------
# get_execution_plan / topological sort
# ---------------------------------------------------------------------------

class TestGetExecutionPlan:
    def test_single_task_plan(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        plan = s.get_execution_plan()
        assert plan == [["a"]]

    def test_independent_tasks_in_one_level(self):
        s = TaskScheduler()
        s.add_task(make_task("a", priority=3), noop)
        s.add_task(make_task("b", priority=7), noop)
        s.add_task(make_task("c", priority=5), noop)
        plan = s.get_execution_plan()
        assert len(plan) == 1
        # Sorted by descending priority within the level
        assert plan[0] == ["b", "c", "a"]

    def test_linear_dependency_chain(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b", dependencies=["a"]), noop)
        s.add_task(make_task("c", dependencies=["b"]), noop)
        plan = s.get_execution_plan()
        assert plan == [["a"], ["b"], ["c"]]

    def test_diamond_dependency(self):
        s = TaskScheduler()
        s.add_task(make_task("root"), noop)
        s.add_task(make_task("left", dependencies=["root"]), noop)
        s.add_task(make_task("right", dependencies=["root"]), noop)
        s.add_task(make_task("tip", dependencies=["left", "right"]), noop)
        plan = s.get_execution_plan()
        assert plan[0] == ["root"]
        assert set(plan[1]) == {"left", "right"}
        assert plan[2] == ["tip"]

    def test_cycle_raises(self):
        # Manually inject a cycle without going through add_task validation
        s = TaskScheduler()
        a = make_task("a", dependencies=["b"])
        b = make_task("b", dependencies=["a"])
        # Bypass add_task to force a cycle
        s._tasks["a"] = a
        s._tasks["b"] = b
        s._handlers["a"] = noop
        s._handlers["b"] = noop
        s.metrics.per_task["a"] = TaskMetrics("a")
        s.metrics.per_task["b"] = TaskMetrics("b")
        with pytest.raises(CircularDependencyError):
            s.get_execution_plan()

    def test_empty_scheduler_plan(self):
        s = TaskScheduler()
        plan = s.get_execution_plan()
        assert plan == []


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_run_single_task(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        metrics = await s.run()
        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "done:a"
        assert metrics.total_time is not None

    @pytest.mark.asyncio
    async def test_run_respects_dependency_order(self):
        order: list[str] = []

        async def recording_handler(task: Task) -> str:
            order.append(task.id)
            return task.id

        s = TaskScheduler()
        s.add_task(make_task("a"), recording_handler)
        s.add_task(make_task("b", dependencies=["a"]), recording_handler)
        s.add_task(make_task("c", dependencies=["b"]), recording_handler)
        await s.run()
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_run_concurrent_independent_tasks(self):
        """Tasks in the same level should overlap when max_concurrency allows."""
        start_times: dict[str, float] = {}
        barrier = asyncio.Event()

        async def barrier_handler(task: Task) -> str:
            start_times[task.id] = time.monotonic()
            await barrier.wait()
            return task.id

        s = TaskScheduler(max_concurrency=4)
        for tid in ("a", "b", "c"):
            s.add_task(make_task(tid), barrier_handler)

        async def release():
            await asyncio.sleep(0.05)
            barrier.set()

        await asyncio.gather(s.run(), release())
        # All three should have started before the barrier was released
        assert len(start_times) == 3

    @pytest.mark.asyncio
    async def test_run_populates_per_task_metrics(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        metrics = await s.run()
        tm = metrics.per_task["a"]
        assert tm.start_time is not None
        assert tm.end_time is not None
        assert tm.elapsed is not None and tm.elapsed >= 0

    @pytest.mark.asyncio
    async def test_run_returns_scheduler_metrics(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        m = await s.run()
        assert isinstance(m, SchedulerMetrics)
        assert m.total_time is not None


# ---------------------------------------------------------------------------
# run() — failure and retry
# ---------------------------------------------------------------------------

class TestRetry:
    @pytest.mark.asyncio
    async def test_task_fails_after_max_retries(self):
        s = TaskScheduler(base_backoff=0.0)
        s.add_task(make_task("a", max_retries=2), failing)
        with pytest.raises(RuntimeError, match="failed after 3 attempt"):
            await s.run()
        assert s.get_task("a").status == TaskStatus.FAILED
        assert s.get_task("a").retry_count == 3

    @pytest.mark.asyncio
    async def test_retry_count_increments(self):
        s = TaskScheduler(base_backoff=0.0)
        s.add_task(make_task("a", max_retries=1), failing)
        with pytest.raises(RuntimeError):
            await s.run()
        assert s.get_task("a").retry_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_called(self):
        """asyncio.sleep should be called with exponentially growing delays."""
        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        s = TaskScheduler(base_backoff=1.0)
        s.add_task(make_task("a", max_retries=2), failing)

        with patch("task_scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(RuntimeError):
                await s.run()

        assert sleep_calls == pytest.approx([1.0, 2.0])

    @pytest.mark.asyncio
    async def test_task_succeeds_on_retry(self):
        attempt = {"count": 0}

        async def flaky(task: Task) -> str:
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise ValueError("not yet")
            return "ok"

        s = TaskScheduler(base_backoff=0.0)
        s.add_task(make_task("a", max_retries=3), flaky)
        await s.run()
        assert s.get_task("a").status == TaskStatus.COMPLETED
        assert s.get_task("a").result == "ok"
        assert attempt["count"] == 3

    @pytest.mark.asyncio
    async def test_failure_aborts_subsequent_levels(self):
        """If a task fails, dependent tasks should not run."""
        ran: list[str] = []

        async def record(task: Task) -> str:
            ran.append(task.id)
            return task.id

        s = TaskScheduler(base_backoff=0.0)
        s.add_task(make_task("a", max_retries=0), failing)
        s.add_task(make_task("b", dependencies=["a"]), record)

        with pytest.raises(RuntimeError):
            await s.run()

        assert "b" not in ran


# ---------------------------------------------------------------------------
# Observer events
# ---------------------------------------------------------------------------

class TestObserverEvents:
    @pytest.mark.asyncio
    async def test_on_task_start_emitted(self):
        started: list[str] = []

        async def on_start(task: Task) -> None:
            started.append(task.id)

        s = TaskScheduler()
        s.subscribe("on_task_start", on_start)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert "a" in started

    @pytest.mark.asyncio
    async def test_on_task_complete_emitted(self):
        completed: list[str] = []

        async def on_complete(task: Task) -> None:
            completed.append(task.id)

        s = TaskScheduler()
        s.subscribe("on_task_complete", on_complete)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert "a" in completed

    @pytest.mark.asyncio
    async def test_on_task_fail_emitted(self):
        failed: list[str] = []

        async def on_fail(task: Task) -> None:
            failed.append(task.id)

        s = TaskScheduler(base_backoff=0.0)
        s.subscribe("on_task_fail", on_fail)
        s.add_task(make_task("a", max_retries=0), failing)

        with pytest.raises(RuntimeError):
            await s.run()

        assert "a" in failed

    @pytest.mark.asyncio
    async def test_on_task_start_receives_correct_task(self):
        received: list[Task] = []

        async def on_start(task: Task) -> None:
            received.append(task)

        s = TaskScheduler()
        s.subscribe("on_task_start", on_start)
        t = make_task("a")
        s.add_task(t, noop)
        await s.run()
        assert received[0].id == "a"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_called(self):
        calls: list[str] = []

        async def cb1(task: Task) -> None:
            calls.append("cb1")

        async def cb2(task: Task) -> None:
            calls.append("cb2")

        s = TaskScheduler()
        s.subscribe("on_task_complete", cb1)
        s.subscribe("on_task_complete", cb2)
        s.add_task(make_task("a"), noop)
        await s.run()
        assert "cb1" in calls
        assert "cb2" in calls


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_get_task_returns_correct_task(self):
        s = TaskScheduler()
        t = make_task("x")
        s.add_task(t, noop)
        assert s.get_task("x") is t

    def test_get_task_raises_for_unknown_id(self):
        s = TaskScheduler()
        with pytest.raises(TaskNotFoundError, match="'missing'"):
            s.get_task("missing")

    def test_get_all_tasks_returns_all(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.add_task(make_task("b"), noop)
        ids = {t.id for t in s.get_all_tasks()}
        assert ids == {"a", "b"}

    def test_get_all_tasks_empty(self):
        s = TaskScheduler()
        assert s.get_all_tasks() == []

    def test_reset_clears_tasks(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.reset()
        assert s.get_all_tasks() == []

    def test_reset_clears_metrics(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.reset()
        assert s.metrics.per_task == {}

    def test_reset_allows_reuse(self):
        s = TaskScheduler()
        s.add_task(make_task("a"), noop)
        s.reset()
        # Should not raise "already registered"
        s.add_task(make_task("a"), noop)
        assert s.get_task("a").id == "a"


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """At most max_concurrency tasks should run at the same time."""
        active = {"count": 0, "max_seen": 0}
        gate = asyncio.Event()

        async def counting_handler(task: Task) -> str:
            active["count"] += 1
            active["max_seen"] = max(active["max_seen"], active["count"])
            await gate.wait()
            active["count"] -= 1
            return task.id

        max_c = 2
        s = TaskScheduler(max_concurrency=max_c)
        for i in range(5):
            s.add_task(make_task(f"t{i}"), counting_handler)

        async def release():
            # Give tasks time to pile up then release
            await asyncio.sleep(0.05)
            gate.set()

        await asyncio.gather(s.run(), release())
        assert active["max_seen"] <= max_c
