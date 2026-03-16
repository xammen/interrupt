"""
task_scheduler.py

Async task scheduler with priority queuing, dependency resolution,
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
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a single schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status of the task.
        retry_count: Number of times the task has been retried so far.
        max_retries: Maximum number of retry attempts allowed.
        created_at: Timestamp of task creation.
        result: The return value produced by the task callable, if any.
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


@dataclass
class TaskMetrics:
    """Per-task execution metrics.

    Attributes:
        task_id: The task this record belongs to.
        start_time: Wall-clock start time (seconds since epoch).
        end_time: Wall-clock end time (seconds since epoch), or None if still running.
        elapsed_seconds: Total wall-clock duration.
        retry_count: Final retry count recorded at completion/failure.
    """

    task_id: str
    start_time: float = 0.0
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    retry_count: int = 0


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a complete scheduler run.

    Attributes:
        total_elapsed_seconds: Wall-clock time from first task start to last task finish.
        task_metrics: Per-task metric records keyed by task ID.
    """

    total_elapsed_seconds: float = 0.0
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observer / event types
# ---------------------------------------------------------------------------

EventHandler = Callable[["Task"], Coroutine[Any, Any, None]]


class _EventBus:
    """Simple async observer bus for scheduler lifecycle events."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        """Register *handler* to be called when *event* is emitted."""
        self._handlers[event].append(handler)

    async def emit(self, event: str, task: "Task") -> None:
        """Call all handlers registered for *event* with *task*."""
        for handler in self._handlers[event]:
            await handler(task)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Features
    --------
    * Priority queue — higher-priority tasks run first within the same
      execution group.
    * Topological sort — resolves dependency graphs before execution;
      independent tasks within a group run concurrently.
    * Circular-dependency detection — raises :class:`CircularDependencyError`.
    * Configurable concurrency limit via ``max_workers``.
    * Exponential-backoff retry logic for failed tasks.
    * Observer pattern events: ``on_task_start``, ``on_task_complete``,
      ``on_task_fail``.
    * :meth:`get_execution_plan` returns ordered groups of task IDs.
    * :attr:`metrics` exposes post-run timing and retry statistics.

    Parameters
    ----------
    max_workers:
        Maximum number of tasks that may run in parallel (default: 4).
    base_backoff_seconds:
        Initial retry delay in seconds; doubles on each subsequent retry
        (default: 1.0).
    """

    def __init__(
        self,
        max_workers: int = 4,
        base_backoff_seconds: float = 1.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._callables: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_workers = max_workers
        self._base_backoff = base_backoff_seconds
        self._event_bus = _EventBus()
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        callable_: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its async callable with the scheduler.

        Parameters
        ----------
        task:
            The :class:`Task` descriptor.
        callable_:
            An *async* callable (coroutine function) that will be awaited
            when the task executes.  It receives no arguments.

        Raises
        ------
        ValueError
            If a task with the same ID has already been registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._callables[task.id] = callable_

    # ------------------------------------------------------------------
    # Event subscription helpers
    # ------------------------------------------------------------------

    def on_task_start(self, handler: EventHandler) -> None:
        """Subscribe *handler* to the ``on_task_start`` event."""
        self._event_bus.subscribe("on_task_start", handler)

    def on_task_complete(self, handler: EventHandler) -> None:
        """Subscribe *handler* to the ``on_task_complete`` event."""
        self._event_bus.subscribe("on_task_complete", handler)

    def on_task_fail(self, handler: EventHandler) -> None:
        """Subscribe *handler* to the ``on_task_fail`` event."""
        self._event_bus.subscribe("on_task_fail", handler)

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all referenced dependency IDs exist.

        Raises
        ------
        TaskNotFoundError
            If a dependency references an unknown task ID.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return tasks grouped into execution layers via Kahn's algorithm.

        Tasks within the same layer have no inter-dependencies and can run
        concurrently.  Within each layer tasks are sorted descending by
        priority so that the highest-priority tasks are presented first.

        Returns
        -------
        List[List[str]]
            Ordered list of groups; each group is a list of task IDs.

        Raises
        ------
        CircularDependencyError
            If a cycle is detected in the dependency graph.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        layers: List[List[str]] = []
        visited_count = 0

        while queue:
            # Collect all zero-in-degree tasks as one concurrent group,
            # sorted by priority descending.
            layer = sorted(
                list(queue),
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
            queue.clear()
            layers.append(layer)
            visited_count += len(layer)

            next_layer_candidates: Set[str] = set()
            for tid in layer:
                for dependent_id in dependents[tid]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_layer_candidates.add(dependent_id)

            queue.extend(next_layer_candidates)

        if visited_count != len(self._tasks):
            # Some tasks were never reachable — there is a cycle.
            cycle_participants = [
                tid for tid, deg in in_degree.items() if deg > 0
            ]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_participants}"
            )

        return layers

    # ------------------------------------------------------------------
    # Public plan introspection
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        Groups are ordered so that earlier groups must complete before later
        groups begin.  Within each group, task IDs are sorted by descending
        priority.

        Returns
        -------
        List[List[str]]
            Execution layers.

        Raises
        ------
        TaskNotFoundError
            If a dependency references an unknown task.
        CircularDependencyError
            If a cycle is detected.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task with retry/backoff logic.

        Acquires *semaphore* to honour the concurrency limit.  On failure,
        retries up to ``task.max_retries`` times with exponential backoff.
        Emits events at each lifecycle transition.

        Parameters
        ----------
        task:
            The task to execute.
        semaphore:
            Shared semaphore that enforces ``max_workers``.
        """
        metrics = self.metrics.task_metrics.setdefault(
            task.id, TaskMetrics(task_id=task.id)
        )

        async with semaphore:
            metrics.start_time = time.monotonic()
            task.status = TaskStatus.RUNNING
            await self._event_bus.emit("on_task_start", task)

            callable_ = self._callables[task.id]
            last_exception: Optional[BaseException] = None

            for attempt in range(task.max_retries + 1):
                try:
                    task.result = await callable_()
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.elapsed_seconds = metrics.end_time - metrics.start_time
                    metrics.retry_count = task.retry_count
                    await self._event_bus.emit("on_task_complete", task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc
                    if attempt < task.max_retries:
                        task.retry_count += 1
                        backoff = self._base_backoff * (2 ** (attempt))
                        await asyncio.sleep(backoff)
                    # else: exhausted retries — fall through

            # All attempts failed.
            task.status = TaskStatus.FAILED
            metrics.end_time = time.monotonic()
            metrics.elapsed_seconds = metrics.end_time - metrics.start_time
            metrics.retry_count = task.retry_count
            await self._event_bus.emit("on_task_fail", task)
            raise RuntimeError(
                f"Task '{task.id}' failed after {task.max_retries} retries."
            ) from last_exception

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and priorities.

        Tasks in the same execution layer run concurrently up to
        ``max_workers``.  If any task exhausts its retries, a
        :class:`RuntimeError` is raised after the current layer completes.

        Returns
        -------
        SchedulerMetrics
            Aggregate and per-task execution metrics.

        Raises
        ------
        TaskNotFoundError
            If a dependency references an unknown task.
        CircularDependencyError
            If a cycle is detected.
        RuntimeError
            If one or more tasks fail after exhausting retries.
        """
        self._validate_dependencies()
        layers = self._topological_sort()

        semaphore = asyncio.Semaphore(self._max_workers)
        scheduler_start = time.monotonic()
        failed_tasks: List[str] = []

        for layer in layers:
            # Skip tasks whose dependencies failed.
            runnable = [
                tid for tid in layer
                if self._tasks[tid].status == TaskStatus.PENDING
            ]

            if not runnable:
                continue

            results = await asyncio.gather(
                *(self._run_task(self._tasks[tid], semaphore) for tid in runnable),
                return_exceptions=True,
            )

            for tid, result in zip(runnable, results):
                if isinstance(result, BaseException):
                    failed_tasks.append(tid)
                    # Mark dependents as failed so they are skipped.
                    self._cascade_failure(tid)

        self.metrics.total_elapsed_seconds = time.monotonic() - scheduler_start

        if failed_tasks:
            raise RuntimeError(
                f"The following tasks failed: {failed_tasks}. "
                "See per-task metrics for details."
            )

        return self.metrics

    def _cascade_failure(self, failed_task_id: str) -> None:
        """Mark all tasks that (transitively) depend on *failed_task_id* as FAILED.

        Parameters
        ----------
        failed_task_id:
            The ID of the task that has already failed.
        """
        for task in self._tasks.values():
            if (
                failed_task_id in task.dependencies
                and task.status == TaskStatus.PENDING
            ):
                task.status = TaskStatus.FAILED
                self._cascade_failure(task.id)
