"""
task_scheduler.py - Async task scheduler with priority queuing, dependency resolution,
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
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority from 1 (lowest) to 10 (highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of times the task has been retried so far.
        max_retries: Maximum number of retry attempts allowed.
        created_at: Timestamp of task creation.
        result: Stored return value after successful execution.
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

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"Priority must be between 1 and 10, got {self.priority}")


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
        """Wall-clock seconds spent executing (excluding retries wait time)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a complete scheduler run."""

    total_start: float = field(default_factory=time.monotonic)
    total_end: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_time(self) -> Optional[float]:
        if self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Observer / Event system
# ---------------------------------------------------------------------------

EventCallback = Callable[["Task"], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async observer that dispatches named events to registered listeners."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Register *callback* for *event*."""
        self._listeners[event].append(callback)

    async def emit(self, event: str, task: "Task") -> None:
        """Invoke all listeners registered for *event* with *task*."""
        for cb in self._listeners[event]:
            await cb(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrent execution.

    Features:
    - Priority queue (higher priority value → earlier execution within a group).
    - Topological sort for dependency resolution; raises on cycles.
    - Configurable concurrency limit via an asyncio.Semaphore.
    - Exponential backoff retry logic for failed tasks.
    - Observer pattern events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.
    - ``get_execution_plan`` returns ordered groups of task IDs.

    Args:
        max_concurrency: Maximum number of tasks that may run simultaneously.
        base_backoff: Base delay in seconds for the first retry.
    """

    def __init__(self, max_concurrency: int = 4, base_backoff: float = 1.0) -> None:
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Callable[["Task"], Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._bus = EventBus()
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        handler: Callable[["Task"], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task together with its async handler coroutine.

        Args:
            task: The :class:`Task` instance to schedule.
            handler: An async callable that accepts the task and returns a result.

        Raises:
            TaskNotFoundError: If any dependency ID in ``task.dependencies`` has
                not been registered yet.
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        for dep_id in task.dependencies:
            if dep_id not in self._tasks:
                raise TaskNotFoundError(
                    f"Dependency '{dep_id}' not found. Register dependencies before dependents."
                )
        self._tasks[task.id] = task
        self._handlers[task.id] = handler
        self.metrics.per_task[task.id] = TaskMetrics(task_id=task.id)

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Forward event subscription to the internal :class:`EventBus`.

        Supported events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.
        """
        self._bus.subscribe(event, callback)

    # ------------------------------------------------------------------
    # Dependency / topology helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> Dict[str, Set[str]]:
        """Return a mapping of task_id → set of task IDs that depend on it."""
        dependents: Dict[str, Set[str]] = defaultdict(set)
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].add(task.id)
        return dependents

    def _topological_sort(self) -> List[List[str]]:
        """Return tasks grouped into execution levels via Kahn's algorithm.

        Each inner list is a batch of task IDs whose dependencies are all
        satisfied by the previous batches.  Within a batch, tasks are ordered
        by descending priority so callers can process them in order.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents = self._build_adjacency()

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] = in_degree.get(task.id, 0)
            in_degree[task.id] += len(task.dependencies)

        # Reset and recompute cleanly
        in_degree = {tid: len(t.dependencies) for tid, t in self._tasks.items()}

        queue: deque[str] = deque(
            sorted(
                (tid for tid, deg in in_degree.items() if deg == 0),
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
        )

        levels: List[List[str]] = []
        visited = 0

        while queue:
            # Drain the current "wave" — all nodes with in_degree == 0
            current_level = list(queue)
            queue.clear()
            # Sort current level by descending priority
            current_level.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            levels.append(current_level)
            visited += len(current_level)

            next_wave: List[str] = []
            for tid in current_level:
                for dependent_id in dependents[tid]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_wave.append(dependent_id)

            next_wave.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            queue.extend(next_wave)

        if visited != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected in the task graph."
            )

        return levels

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution plan as groups of task IDs.

        Each group can be executed concurrently; groups must be executed
        sequentially in the order returned.

        Returns:
            A list of lists of task IDs ordered by dependency level and
            descending priority within each level.

        Raises:
            CircularDependencyError: On cyclic dependencies.
        """
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task) -> None:
        """Execute a single task with retry and backoff logic.

        Emits ``on_task_start`` before execution, ``on_task_complete`` on
        success, and ``on_task_fail`` after all retries are exhausted.

        The task's ``status`` and ``result`` fields are updated in-place.
        Metrics (``start_time``, ``end_time``, ``retry_count``) are recorded
        in :attr:`metrics`.
        """
        tm = self.metrics.per_task[task.id]
        handler = self._handlers[task.id]

        task.status = TaskStatus.RUNNING
        await self._bus.emit("on_task_start", task)
        tm.start_time = time.monotonic()

        last_exc: Optional[BaseException] = None

        for attempt in range(task.max_retries + 1):
            try:
                async with self._semaphore:
                    task.result = await handler(task)
                task.status = TaskStatus.COMPLETED
                tm.end_time = time.monotonic()
                tm.retry_count = task.retry_count
                await self._bus.emit("on_task_complete", task)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                task.retry_count += 1
                tm.retry_count = task.retry_count
                if attempt < task.max_retries:
                    backoff = self._base_backoff * (2 ** attempt)
                    await asyncio.sleep(backoff)

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        tm.end_time = time.monotonic()
        await self._bus.emit("on_task_fail", task)
        raise RuntimeError(
            f"Task '{task.id}' failed after {task.max_retries + 1} attempt(s)."
        ) from last_exc

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed level by level (dependency order).  Within each
        level all tasks are launched concurrently, bounded by
        ``max_concurrency``.  If any task in a level fails after all retries
        the remaining levels are **not** executed and the exception propagates.

        Returns:
            The populated :class:`SchedulerMetrics` instance.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            RuntimeError: If any task exhausts its retry budget.
        """
        # Preserve per-task entries already created by add_task; only reset timing.
        existing_per_task = {
            tid: TaskMetrics(task_id=tid) for tid in self._tasks
        }
        self.metrics = SchedulerMetrics(per_task=existing_per_task)
        levels = self._topological_sort()

        try:
            for level in levels:
                tasks_in_level = [self._tasks[tid] for tid in level]
                # Reset semaphore for current concurrency setting (already set in __init__)
                await asyncio.gather(
                    *(self._run_task(t) for t in tasks_in_level)
                )
        finally:
            self.metrics.total_end = time.monotonic()

        return self.metrics

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` registered under *task_id*.

        Raises:
            TaskNotFoundError: If *task_id* is not registered.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task with id '{task_id}'.")

    def get_all_tasks(self) -> List[Task]:
        """Return all registered tasks in registration order."""
        return list(self._tasks.values())

    def reset(self) -> None:
        """Clear all registered tasks, handlers, and metrics.

        Useful when reusing a scheduler instance across multiple runs.
        """
        self._tasks.clear()
        self._handlers.clear()
        self.metrics = SchedulerMetrics()
