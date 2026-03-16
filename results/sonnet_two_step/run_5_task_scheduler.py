"""
task_scheduler.py

Async task scheduler with priority queue, dependency resolution,
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


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist."""


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique task identifier.
        name: Human-readable task name.
        priority: Scheduling priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task.
        status: Current lifecycle status of the task.
        retry_count: Number of times the task has been retried.
        max_retries: Maximum number of retry attempts before marking as FAILED.
        created_at: Timestamp when the task was created.
        result: Return value produced by the task callable, if any.
    """

    id: str
    name: str
    priority: int  # 1–10
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Per-task execution timing information."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds from start to finish, or None if not finished."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate scheduler execution metrics."""

    total_start_time: Optional[float] = None
    total_end_time: Optional[float] = None
    tasks: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_start_time is not None and self.total_end_time is not None:
            return self.total_end_time - self.total_start_time
        return None


# ---------------------------------------------------------------------------
# Observer / event system
# ---------------------------------------------------------------------------

EventHandler = Callable[["Task"], Coroutine[Any, Any, None]]


class EventEmitter:
    """Lightweight async observer/event emitter."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register *handler* for *event*."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Deregister *handler* from *event*."""
        self._handlers[event] = [h for h in self._handlers[event] if h is not handler]

    async def emit(self, event: str, task: "Task") -> None:
        """Fire all handlers registered for *event*, passing *task*."""
        for handler in list(self._handlers[event]):
            await handler(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async priority task scheduler with dependency resolution.

    Features
    --------
    - Topological sort for dependency ordering with cycle detection.
    - Concurrent execution of independent tasks (configurable limit).
    - Exponential back-off retry logic.
    - Observer pattern events: ``on_task_start``, ``on_task_complete``,
      ``on_task_fail``.
    - Execution plan generation and detailed metrics tracking.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks allowed to run in parallel (default: 4).
    base_retry_delay:
        Base delay in seconds for exponential back-off (default: 1.0).
    """

    _EVENT_START = "on_task_start"
    _EVENT_COMPLETE = "on_task_complete"
    _EVENT_FAIL = "on_task_fail"

    def __init__(
        self,
        max_concurrency: int = 4,
        base_retry_delay: float = 1.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._callables: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_retry_delay = base_retry_delay
        self._emitter = EventEmitter()
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------

    def register(
        self,
        task: Task,
        coro_fn: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its async callable.

        Parameters
        ----------
        task:
            The :class:`Task` descriptor.
        coro_fn:
            An *async* callable (coroutine function) invoked when the task
            runs.  It receives the :class:`Task` instance as its sole
            argument.
        """
        self._tasks[task.id] = task
        self._callables[task.id] = coro_fn
        self.metrics.tasks[task.id] = TaskMetrics(task_id=task.id)

    # ------------------------------------------------------------------
    # Observer registration helpers
    # ------------------------------------------------------------------

    def on_task_start(self, handler: EventHandler) -> None:
        """Register a handler called when a task begins execution."""
        self._emitter.on(self._EVENT_START, handler)

    def on_task_complete(self, handler: EventHandler) -> None:
        """Register a handler called when a task completes successfully."""
        self._emitter.on(self._EVENT_COMPLETE, handler)

    def on_task_fail(self, handler: EventHandler) -> None:
        """Register a handler called when a task exhausts all retries."""
        self._emitter.on(self._EVENT_FAIL, handler)

    # ------------------------------------------------------------------
    # Dependency graph helpers
    # ------------------------------------------------------------------

    def _validate_task_ids(self) -> None:
        """Ensure all dependency IDs reference registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _topological_sort(self) -> List[List[Task]]:
        """Return tasks grouped into independent execution levels.

        Each group contains tasks that can run concurrently once all
        tasks in previous groups have completed.

        Raises
        ------
        CircularDependencyError
            If a cycle exists in the dependency graph.
        """
        self._validate_task_ids()

        # Build in-degree map and adjacency list
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Kahn's algorithm – level by level
        levels: List[List[Task]] = []
        current_level = [tid for tid, deg in in_degree.items() if deg == 0]

        visited = 0
        while current_level:
            # Sort within a level by descending priority for deterministic order
            level_tasks = sorted(
                (self._tasks[tid] for tid in current_level),
                key=lambda t: t.priority,
                reverse=True,
            )
            levels.append(level_tasks)
            next_level: List[str] = []
            for task in level_tasks:
                visited += 1
                for dependent_id in dependents[task.id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_level.append(dependent_id)
            current_level = next_level

        if visited != len(self._tasks):
            cycle_ids = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_ids}"
            )

        return levels

    # ------------------------------------------------------------------
    # Public plan API
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[Task]]:
        """Return the ordered execution groups without running anything.

        Returns
        -------
        List[List[Task]]
            Each inner list is a group of tasks that may execute in
            parallel.  Groups are ordered from first to last.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        TaskNotFoundError
            If any dependency references an unknown task ID.
        """
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution core
    # ------------------------------------------------------------------

    async def _run_task_with_retry(self, task: Task) -> None:
        """Execute *task* with exponential back-off retry logic."""
        m = self.metrics.tasks[task.id]
        attempt = 0

        while True:
            task.status = TaskStatus.RUNNING
            m.start_time = time.monotonic()
            await self._emitter.emit(self._EVENT_START, task)

            try:
                coro_fn = self._callables[task.id]
                task.result = await coro_fn(task)
                task.status = TaskStatus.COMPLETED
                m.end_time = time.monotonic()
                m.retry_count = task.retry_count
                await self._emitter.emit(self._EVENT_COMPLETE, task)
                return

            except Exception:  # noqa: BLE001
                attempt += 1
                task.retry_count += 1
                m.retry_count = task.retry_count

                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    m.end_time = time.monotonic()
                    await self._emitter.emit(self._EVENT_FAIL, task)
                    return

                # Exponential back-off: base * 2^attempt
                delay = self._base_retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
                task.status = TaskStatus.PENDING

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting priorities and dependencies.

        Tasks within the same dependency level run concurrently up to
        ``max_concurrency``.  A failed task does *not* block execution of
        unrelated tasks, but any task that depends on a failed task is
        skipped (marked FAILED immediately).

        Returns
        -------
        SchedulerMetrics
            Populated metrics object with per-task and total timings.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        TaskNotFoundError
            If any dependency references an unknown task ID.
        """
        levels = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        self.metrics.total_start_time = time.monotonic()

        # Track which task IDs have failed so dependents can be skipped
        failed_ids: Set[str] = set()

        for level in levels:
            # Partition tasks: skip those whose dependencies failed
            runnable: List[Task] = []
            for task in level:
                if any(dep in failed_ids for dep in task.dependencies):
                    task.status = TaskStatus.FAILED
                    failed_ids.add(task.id)
                    m = self.metrics.tasks[task.id]
                    m.start_time = time.monotonic()
                    m.end_time = m.start_time
                    await self._emitter.emit(self._EVENT_FAIL, task)
                else:
                    runnable.append(task)

            async def _bounded(t: Task) -> None:
                async with semaphore:
                    await self._run_task_with_retry(t)

            await asyncio.gather(*(_bounded(t) for t in runnable))

            # Collect any newly failed tasks
            for task in runnable:
                if task.status == TaskStatus.FAILED:
                    failed_ids.add(task.id)

        self.metrics.total_end_time = time.monotonic()
        return self.metrics

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` with the given *task_id*.

        Raises
        ------
        TaskNotFoundError
            If *task_id* is not registered.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task registered with id '{task_id}'")

    @property
    def tasks(self) -> Dict[str, Task]:
        """Read-only view of all registered tasks keyed by ID."""
        return dict(self._tasks)
