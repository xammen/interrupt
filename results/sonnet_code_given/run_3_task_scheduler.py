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
            handler: An async callable ``async def handler(task: Task) -> Any``
                     that performs the actual work.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._handlers[task.id] = handler

    def on_task_start(self, callback: EventCallback) -> None:
        """Subscribe *callback* to the task-start event."""
        self._bus.subscribe("on_task_start", callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Subscribe *callback* to the task-complete event."""
        self._bus.subscribe("on_task_complete", callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Subscribe *callback* to the task-fail event."""
        self._bus.subscribe("on_task_fail", callback)

    # ------------------------------------------------------------------
    # Dependency / topology helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency refers to a registered task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def get_execution_plan(self) -> List[List[str]]:
        """Compute ordered execution groups via Kahn's topological sort.

        Tasks within the same group have no inter-dependencies and can run
        concurrently.  Within each group tasks are sorted by descending
        priority (higher value first).

        Returns:
            A list of groups; each group is a list of task IDs.

        Raises:
            CircularDependencyError: If a cycle is detected.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()

        # Build in-degree map and adjacency list (dep → dependents)
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Kahn's BFS
        queue: deque[str] = deque(
            sorted(
                (tid for tid, deg in in_degree.items() if deg == 0),
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
        )
        groups: List[List[str]] = []
        visited = 0

        while queue:
            # Drain current level as one execution group
            group_size = len(queue)
            group: List[str] = []
            for _ in range(group_size):
                tid = queue.popleft()
                group.append(tid)
                visited += 1
                for dep in dependents[tid]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

            # Sort group by descending priority
            group.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(group)

        if visited != len(self._tasks):
            # Some nodes were never reached → cycle exists
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task) -> None:
        """Execute a single task with retry / backoff logic.

        Acquires the concurrency semaphore before starting, releases on finish.
        Emits ``on_task_start``, ``on_task_complete``, or ``on_task_fail`` events.
        """
        metrics = self.metrics.per_task.setdefault(task.id, TaskMetrics(task_id=task.id))
        handler = self._handlers[task.id]

        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            metrics.start_time = time.monotonic()
            await self._bus.emit("on_task_start", task)

            last_exc: Optional[Exception] = None
            attempt = 0

            while attempt <= task.max_retries:
                try:
                    task.result = await handler(task)
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    await self._bus.emit("on_task_complete", task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    attempt += 1
                    task.retry_count += 1
                    if attempt <= task.max_retries:
                        backoff = self._base_backoff * (2 ** (attempt - 1))
                        await asyncio.sleep(backoff)

            # Exhausted retries
            task.status = TaskStatus.FAILED
            metrics.end_time = time.monotonic()
            metrics.retry_count = task.retry_count
            await self._bus.emit("on_task_fail", task)
            raise RuntimeError(
                f"Task '{task.id}' failed after {task.retry_count} retries."
            ) from last_exc

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Each execution group (layer of the topological sort) is awaited fully
        before the next group begins.  Within a group, tasks run concurrently
        up to ``max_concurrency``.

        Returns:
            A :class:`SchedulerMetrics` instance with timing data.

        Raises:
            CircularDependencyError: Propagated from :meth:`get_execution_plan`.
            RuntimeError: If any task exhausts its retry budget.
        """
        self.metrics = SchedulerMetrics()
        plan = self.get_execution_plan()

        for group in plan:
            await asyncio.gather(
                *[self._run_task(self._tasks[tid]) for tid in group],
                return_exceptions=False,
            )

        self.metrics.total_end = time.monotonic()
        return self.metrics
