"""
task_scheduler.py - Async task scheduler with priority queues, dependency resolution,
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
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id:           Unique identifier for the task.
        name:         Human-readable task name.
        priority:     Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status:       Current lifecycle status of the task.
        retry_count:  Number of times the task has been retried so far.
        max_retries:  Maximum allowed retries before marking the task FAILED.
        created_at:   Timestamp when the task was created.
        result:       Value returned by the task coroutine on success.
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
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a scheduler run."""

    total_start: float = field(default_factory=time.monotonic)
    total_end: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Observer / event types
# ---------------------------------------------------------------------------

EventCallback = Callable[[Task], Coroutine[Any, Any, None]]


class _EventBus:
    """Lightweight async event bus for task lifecycle hooks."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event: str, callback: EventCallback) -> None:
        self._listeners[event].append(callback)

    async def emit(self, event: str, task: Task) -> None:
        for cb in self._listeners[event]:
            await cb(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrent execution.

    Features:
    - Priority-based scheduling (higher priority = earlier execution).
    - Topological sort for dependency resolution.
    - Circular dependency detection.
    - Configurable concurrency limit (semaphore-backed).
    - Exponential backoff retry logic.
    - Observer pattern: on_task_start, on_task_complete, on_task_fail events.
    - Execution plan introspection via :meth:`get_execution_plan`.
    - Detailed per-task and aggregate timing metrics.

    Args:
        max_concurrency: Maximum number of tasks allowed to run in parallel.
        base_backoff:    Base delay (seconds) for exponential backoff on retry.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        base_backoff: float = 1.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff
        self._bus = _EventBus()
        self._metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its async coroutine factory with the scheduler.

        Args:
            task:               The :class:`Task` descriptor.
            coroutine_factory:  A zero-argument callable that returns a fresh
                                coroutine each time it is called (needed for retries).

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_factory
        self._metrics.per_task[task.id] = TaskMetrics(task_id=task.id)

    def on_task_start(self, callback: EventCallback) -> None:
        """Subscribe to the task-start event."""
        self._bus.subscribe("task_start", callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Subscribe to the task-complete event."""
        self._bus.subscribe("task_complete", callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Subscribe to the task-fail event."""
        self._bus.subscribe("task_fail", callback)

    # ------------------------------------------------------------------
    # Dependency / topology helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all referenced dependency IDs exist."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return tasks grouped into execution waves via Kahn's algorithm.

        Each wave contains task IDs that can run concurrently (all their
        dependencies are satisfied by previous waves).

        Returns:
            Ordered list of groups; each group is a list of task IDs
            sorted by descending priority within the group.

        Raises:
            CircularDependencyError: If a cycle is detected.
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
        waves: List[List[str]] = []
        processed = 0

        while queue:
            # Collect the current wave (all zero-in-degree nodes)
            wave_size = len(queue)
            wave: List[str] = []
            for _ in range(wave_size):
                tid = queue.popleft()
                wave.append(tid)
                processed += 1
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            # Sort wave by priority descending (10 first)
            wave.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            waves.append(wave)

        if processed != len(self._tasks):
            # Some nodes were never enqueued — cycle exists
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return waves

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Returns:
            List of task-ID groups as produced by the topological sort.

        Raises:
            TaskNotFoundError:       If any dependency references an unknown task.
            CircularDependencyError: If a cycle exists in the dependency graph.
        """
        self._validate_dependencies()
        return self._topological_sort()

    @property
    def metrics(self) -> SchedulerMetrics:
        """Read-only access to execution metrics."""
        return self._metrics

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    async def _run_task_with_retry(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task under the concurrency semaphore with retries.

        Uses exponential backoff: delay = base_backoff * 2^(retry_count - 1).
        """
        async with semaphore:
            task.status = TaskStatus.RUNNING
            tm = self._metrics.per_task[task.id]
            tm.start_time = time.monotonic()
            await self._bus.emit("task_start", task)

            while True:
                try:
                    coro = self._coroutines[task.id]()
                    task.result = await coro
                    task.status = TaskStatus.COMPLETED
                    tm.end_time = time.monotonic()
                    tm.retry_count = task.retry_count
                    await self._bus.emit("task_complete", task)
                    return

                except Exception:  # noqa: BLE001
                    task.retry_count += 1
                    tm.retry_count = task.retry_count

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        tm.end_time = time.monotonic()
                        await self._bus.emit("task_fail", task)
                        return

                    backoff = self._base_backoff * (2 ** (task.retry_count - 1))
                    await asyncio.sleep(backoff)

    async def run(self) -> Dict[str, Task]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            Mapping of task ID → completed :class:`Task` objects.

        Raises:
            TaskNotFoundError:       If any dependency references an unknown task.
            CircularDependencyError: If a cycle exists in the dependency graph.
        """
        self._validate_dependencies()
        waves = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics.total_start = time.monotonic()

        for wave in waves:
            # Filter out tasks whose dependencies failed
            runnable = []
            for tid in wave:
                task = self._tasks[tid]
                dep_failed = any(
                    self._tasks[dep].status == TaskStatus.FAILED
                    for dep in task.dependencies
                )
                if dep_failed:
                    task.status = TaskStatus.FAILED
                    self._metrics.per_task[tid].retry_count = task.retry_count
                else:
                    runnable.append(task)

            await asyncio.gather(
                *(self._run_task_with_retry(t, semaphore) for t in runnable)
            )

        self._metrics.total_end = time.monotonic()
        return dict(self._tasks)
