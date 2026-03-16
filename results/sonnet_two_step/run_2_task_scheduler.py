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


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum number of retry attempts before marking as FAILED.
        created_at: Timestamp of task creation.
        result: Optional result value set upon successful completion.
        fn: Async callable that performs the actual work.
    """

    id: str
    name: str
    priority: int  # 1-10
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Any] = None
    fn: Optional[Callable[[], Coroutine[Any, Any, Any]]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if not (1 <= self.priority <= 10):
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------


@dataclass
class TaskMetrics:
    """Per-task execution metrics."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds spent executing (excluding retry delays)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a scheduler run."""

    total_start: Optional[float] = None
    total_end: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_start is not None and self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Observer / event system
# ---------------------------------------------------------------------------

EventCallback = Callable[[Task], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async observer/event bus."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Register *callback* to be called when *event* is emitted."""
        self._listeners[event].append(callback)

    async def emit(self, event: str, task: Task) -> None:
        """Emit *event* and await all registered callbacks."""
        for cb in self._listeners[event]:
            await cb(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Features
    --------
    * Priority queue — higher-priority tasks run first within the same group.
    * Dependency graph — topological sort ensures prerequisites finish first.
    * Circular-dependency detection — raises :class:`CircularDependencyError`.
    * Concurrent execution — independent tasks run in parallel up to
      *max_concurrency*.
    * Exponential backoff retries — failed tasks are retried with increasing
      delays until *max_retries* is exhausted.
    * Observer pattern — subscribe to ``on_task_start``, ``on_task_complete``,
      and ``on_task_fail`` events.
    * Execution plan — :meth:`get_execution_plan` returns ordered groups of
      task IDs that can run concurrently.
    * Metrics — wall-clock times per task and overall, plus retry counts.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks allowed to run simultaneously (default: 4).
    """

    ON_TASK_START = "on_task_start"
    ON_TASK_COMPLETE = "on_task_complete"
    ON_TASK_FAIL = "on_task_fail"

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: Dict[str, Task] = {}
        self._max_concurrency = max_concurrency
        self._bus = EventBus()
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Public registration helpers
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        """Register *task* with the scheduler.

        Parameters
        ----------
        task:
            A :class:`Task` instance to be scheduled.

        Raises
        ------
        ValueError
            If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Subscribe *callback* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.

        Parameters
        ----------
        event:
            Event name string.
        callback:
            Async callable that receives the relevant :class:`Task`.
        """
        self._bus.subscribe(event, callback)

    # ------------------------------------------------------------------
    # Dependency / graph helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency references a known task ID.

        Raises
        ------
        ValueError
            If an unknown dependency is referenced.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Compute execution groups via Kahn's algorithm.

        Each group contains task IDs that can be executed concurrently once
        all tasks in preceding groups have completed.

        Returns
        -------
        List[List[str]]
            Ordered list of execution groups. Within each group tasks are
            sorted by descending priority (highest priority first).

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
        groups: List[List[str]] = []
        processed = 0

        while queue:
            # Collect the current "wave" of ready tasks
            wave = list(queue)
            queue.clear()

            # Sort by descending priority within the wave
            wave.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(wave)
            processed += len(wave)

            next_ready: Set[str] = set()
            for tid in wave:
                for dependent_id in dependents[tid]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_ready.add(dependent_id)

            for tid in sorted(
                next_ready,
                key=lambda t: self._tasks[t].priority,
                reverse=True,
            ):
                queue.append(tid)

        if processed != len(self._tasks):
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution plan as groups of task IDs.

        Each inner list represents tasks that can run concurrently.
        Groups are ordered so that earlier groups must finish before later
        ones begin.

        Returns
        -------
        List[List[str]]
            Execution plan.

        Raises
        ------
        CircularDependencyError
            If a circular dependency is detected.
        ValueError
            If an unknown dependency is referenced.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task_with_retry(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute *task* under *semaphore*, retrying on failure.

        Uses exponential backoff: ``delay = 2 ** retry_count`` seconds.

        Parameters
        ----------
        task:
            The task to execute.
        semaphore:
            Shared concurrency limiter.
        """
        metrics = self.metrics.per_task.setdefault(
            task.id, TaskMetrics(task_id=task.id)
        )

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                metrics.start_time = time.monotonic()
                await self._bus.emit(self.ON_TASK_START, task)

                try:
                    if task.fn is None:
                        raise RuntimeError(
                            f"Task '{task.id}' has no callable assigned (fn=None)."
                        )
                    task.result = await task.fn()
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    await self._bus.emit(self.ON_TASK_COMPLETE, task)
                    return

                except Exception:  # noqa: BLE001
                    metrics.end_time = time.monotonic()
                    task.retry_count += 1
                    metrics.retry_count = task.retry_count

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        await self._bus.emit(self.ON_TASK_FAIL, task)
                        return

                    # Exponential backoff before next attempt
                    backoff = 2 ** (task.retry_count - 1)
                    await asyncio.sleep(backoff)
                    # Reset start time for the next attempt
                    metrics.start_time = time.monotonic()

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed group-by-group (topological order). Within each
        group, up to *max_concurrency* tasks run concurrently.

        Returns
        -------
        SchedulerMetrics
            Collected timing and retry metrics.

        Raises
        ------
        CircularDependencyError
            If a circular dependency is detected before execution begins.
        ValueError
            If an unknown dependency is referenced.
        """
        self._validate_dependencies()
        groups = self._topological_sort()

        semaphore = asyncio.Semaphore(self._max_concurrency)
        self.metrics = SchedulerMetrics()
        self.metrics.total_start = time.monotonic()

        for group in groups:
            # Reset status for tasks that may have been re-submitted
            tasks_in_group = [self._tasks[tid] for tid in group]
            await asyncio.gather(
                *(
                    self._run_task_with_retry(task, semaphore)
                    for task in tasks_in_group
                )
            )

        self.metrics.total_end = time.monotonic()
        return self.metrics
