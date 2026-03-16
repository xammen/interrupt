"""
task_scheduler.py

Async task scheduler with priority queues, dependency resolution,
concurrent execution, retry logic, and observer pattern event emission.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


class DuplicateTaskError(Exception):
    """Raised when a task with an existing ID is added to the scheduler."""


class UnknownDependencyError(Exception):
    """Raised when a task declares a dependency on an unknown task ID."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable label.
        priority: Scheduling priority in the range [1, 10].  Higher values
            are executed first when dependencies allow.
        dependencies: List of task IDs that must complete before this task
            may be started.
        status: Current lifecycle state of the task.
        retry_count: Number of times the task has been retried after failure.
        max_retries: Maximum number of retry attempts before the task is
            permanently marked as FAILED.
        created_at: Timestamp of task creation.
        result: Stores the return value of the task coroutine on success, or
            the exception on permanent failure.
    """

    id: str
    name: str
    priority: int  # 1–10
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = field(default=TaskStatus.PENDING)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class TaskMetrics:
    """Execution metrics for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds spent executing the task (excluding wait time)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a complete scheduler run."""

    total_start_time: Optional[float] = None
    total_end_time: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_start_time is not None and self.total_end_time is not None:
            return self.total_end_time - self.total_start_time
        return None


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EventCallback = Callable[["Task"], Coroutine[Any, Any, None]]


class _EventKind(Enum):
    TASK_START = "on_task_start"
    TASK_COMPLETE = "on_task_complete"
    TASK_FAIL = "on_task_fail"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrent execution.

    Features
    --------
    * Priority queue – higher-priority tasks run first within each
      topological layer.
    * Dependency resolution via Kahn's topological sort algorithm.
    * Circular-dependency detection raising :exc:`CircularDependencyError`.
    * Concurrent execution of independent tasks bounded by *max_concurrency*.
    * Exponential back-off retry logic for transiently failing tasks.
    * Observer pattern: register coroutine callbacks for task lifecycle events.
    * :meth:`get_execution_plan` returns ordered execution groups without
      running anything.
    * :attr:`metrics` exposes per-task and aggregate timing data after a run.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks that may run in parallel. ``0`` means
        unlimited.
    base_retry_delay:
        Base delay (seconds) for the first retry. Subsequent retries use
        exponential back-off: ``base_retry_delay * 2 ** attempt``.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        base_retry_delay: float = 1.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_retry_delay = base_retry_delay
        self._listeners: Dict[_EventKind, List[EventCallback]] = defaultdict(list)
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and the coroutine factory that implements its work.

        Parameters
        ----------
        task:
            Fully configured :class:`Task` instance.
        coroutine_factory:
            A zero-argument callable that returns a fresh coroutine each time
            it is invoked.  It will be called once per attempt (including
            retries).

        Raises
        ------
        DuplicateTaskError
            If a task with the same ID has already been registered.
        """
        if task.id in self._tasks:
            raise DuplicateTaskError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_factory

    # ------------------------------------------------------------------
    # Observer / event API
    # ------------------------------------------------------------------

    def on_task_start(self, callback: EventCallback) -> None:
        """Register a coroutine callback invoked when a task begins execution."""
        self._listeners[_EventKind.TASK_START].append(callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Register a coroutine callback invoked when a task finishes successfully."""
        self._listeners[_EventKind.TASK_COMPLETE].append(callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Register a coroutine callback invoked when a task fails permanently."""
        self._listeners[_EventKind.TASK_FAIL].append(callback)

    async def _emit(self, kind: _EventKind, task: Task) -> None:
        for cb in self._listeners[kind]:
            await cb(task)

    # ------------------------------------------------------------------
    # Dependency graph helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all declared dependencies reference registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise UnknownDependencyError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return execution groups via Kahn's algorithm.

        Each group is a list of task IDs that can run concurrently (all their
        dependencies have been satisfied by preceding groups).  Within a group
        tasks are sorted by descending priority.

        Returns
        -------
        List[List[str]]
            Ordered execution groups, e.g. ``[["a"], ["b", "c"], ["d"]]``.

        Raises
        ------
        CircularDependencyError
            When the dependency graph contains a cycle.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Seed queue with tasks that have no dependencies
        ready: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: List[List[str]] = []
        visited = 0

        while ready:
            # Collect all currently ready tasks as one concurrent group,
            # sorted by descending priority.
            current_group = sorted(
                ready,
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
            ready.clear()
            groups.append(current_group)
            visited += len(current_group)

            next_ready: List[str] = []
            for tid in current_group:
                for dependent_id in dependents[tid]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_ready.append(dependent_id)

            ready.extend(next_ready)

        if visited != len(self._tasks):
            # Some tasks were never reached → cycle exists
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Public planning API
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution plan without running any tasks.

        Returns
        -------
        List[List[str]]
            Each inner list is a group of task IDs that may execute in
            parallel.  Groups are ordered such that all dependencies of a
            group are satisfied by preceding groups.

        Raises
        ------
        UnknownDependencyError
            If any dependency references a task that has not been registered.
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task_with_retry(self, task: Task) -> None:
        """Execute *task* with exponential back-off retries.

        Updates task.status, task.result, and task.retry_count in place.
        Emits TASK_START on the first attempt and TASK_COMPLETE / TASK_FAIL
        when the task definitively finishes.
        """
        m = self.metrics.per_task.setdefault(task.id, TaskMetrics(task_id=task.id))
        factory = self._coroutines[task.id]

        task.status = TaskStatus.RUNNING
        m.start_time = time.monotonic()
        await self._emit(_EventKind.TASK_START, task)

        last_exc: Optional[BaseException] = None

        for attempt in range(task.max_retries + 1):
            try:
                task.result = await factory()
                task.status = TaskStatus.COMPLETED
                m.end_time = time.monotonic()
                m.retry_count = task.retry_count
                await self._emit(_EventKind.TASK_COMPLETE, task)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < task.max_retries:
                    task.retry_count += 1
                    delay = self._base_retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                # else: fall through to permanent failure

        task.status = TaskStatus.FAILED
        task.result = last_exc
        m.end_time = time.monotonic()
        m.retry_count = task.retry_count
        await self._emit(_EventKind.TASK_FAIL, task)

    async def run(self) -> Dict[str, Task]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks within the same topological group run concurrently, bounded by
        *max_concurrency*.  If a task fails permanently its dependents are
        still attempted (they will receive ``FAILED`` dependencies, which
        callers can inspect via ``task.result``).

        Returns
        -------
        Dict[str, Task]
            Mapping of task ID → completed :class:`Task` instance.

        Raises
        ------
        UnknownDependencyError
            If any dependency references an unregistered task.
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        groups = self._topological_sort()

        semaphore = (
            asyncio.Semaphore(self._max_concurrency)
            if self._max_concurrency > 0
            else None
        )

        self.metrics = SchedulerMetrics()
        self.metrics.total_start_time = time.monotonic()

        async def _guarded(task: Task) -> None:
            if semaphore is not None:
                async with semaphore:
                    await self._run_task_with_retry(task)
            else:
                await self._run_task_with_retry(task)

        for group in groups:
            await asyncio.gather(*[_guarded(self._tasks[tid]) for tid in group])

        self.metrics.total_end_time = time.monotonic()
        return dict(self._tasks)
