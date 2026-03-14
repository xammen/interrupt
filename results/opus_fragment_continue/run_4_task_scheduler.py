"""
Async task scheduler with priority queue, dependency resolution,
retry logic, concurrency control, and observer-pattern event emission.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        self.cycle = cycle or []
        detail = f": {' -> '.join(self.cycle)}" if self.cycle else ""
        super().__init__(f"Circular dependency detected{detail}")


class TaskNotFoundError(Exception):
    """Raised when a referenced task does not exist."""


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work to be scheduled.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of retries already attempted.
        max_retries: Maximum number of retry attempts before permanent failure.
        created_at: Timestamp when the task was created.
        result: The return value of the coroutine, or ``None``.
    """

    id: str
    name: str
    priority: int = 5
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Timing and retry metrics for a single task execution."""
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    retries: int = 0

    @property
    def duration(self) -> float:
        """Wall-clock duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for an entire scheduler run."""
    total_time: float = 0.0
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observer / event types
# ---------------------------------------------------------------------------

# Callback signatures accepted by the observer system.
EventCallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._metrics = SchedulerMetrics()
        self._semaphore: asyncio.Semaphore | None = None

    # -- Task registration ---------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The :class:`Task` descriptor.
            coro: An async callable that performs the work.  It will be
                  called with no arguments.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    # -- Observer pattern -----------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> None:
        """Subscribe *callback* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.
        """
        self._listeners[event].append(callback)

    async def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all listeners registered for *event*."""
        for cb in self._listeners.get(event, []):
            ret = cb(*args, **kwargs)
            if asyncio.iscoroutine(ret):
                await ret

    # -- Dependency graph utilities -------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency reference points to a known task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_cycles(self) -> None:
        """Detect circular dependencies using iterative DFS.

        Raises:
            CircularDependencyError: If a cycle is found, with the cycle path.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        for start in self._tasks:
            if color[start] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            color[start] = GRAY
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx < len(deps):
                    # Push current node back with incremented index
                    stack.append((node, idx + 1))
                    dep = deps[idx]
                    if color[dep] == GRAY:
                        # Reconstruct cycle path
                        cycle = [dep, node]
                        for n, _ in reversed(stack):
                            if n == dep:
                                break
                            cycle.append(n)
                        cycle.reverse()
                        raise CircularDependencyError(cycle)
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        parent[dep] = node
                        stack.append((dep, 0))
                else:
                    color[node] = BLACK

    def _topological_sort(self) -> list[list[str]]:
        """Return tasks grouped into execution levels (Kahn's algorithm).

        Each level is a list of task IDs whose dependencies are all
        satisfied by tasks in earlier levels.  Within a level, tasks
        are sorted by descending priority so that higher-priority work
        is started first.

        Returns:
            A list of levels, where each level is a list of task IDs.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Seed with all zero-in-degree tasks, sorted by priority (desc)
        queue: list[str] = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda tid: self._tasks[tid].priority,
            reverse=True,
        )

        levels: list[list[str]] = []
        while queue:
            levels.append(queue)
            next_queue: list[str] = []
            for tid in queue:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            next_queue.sort(
                key=lambda tid: self._tasks[tid].priority, reverse=True
            )
            queue = next_queue

        return levels

    # -- Task execution -------------------------------------------------------

    async def _execute_task(self, task_id: str) -> None:
        """Execute a single task with retry logic and metrics collection.

        The method respects the concurrency semaphore, emits lifecycle
        events, and records timing information.
        """
        assert self._semaphore is not None
        task = self._tasks[task_id]
        coro_fn = self._coroutines[task_id]
        metrics = TaskMetrics(task_id=task_id)

        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            await self._emit("on_task_start", task)
            metrics.start_time = time.monotonic()

            last_exc: BaseException | None = None
            for attempt in range(task.max_retries + 1):
                try:
                    task.result = await coro_fn()
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retries = attempt
                    task.retry_count = attempt
                    self._metrics.task_metrics[task_id] = metrics
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    last_exc = exc
                    task.retry_count = attempt + 1

            # All retries exhausted
            task.status = TaskStatus.FAILED
            metrics.end_time = time.monotonic()
            metrics.retries = task.max_retries
            self._metrics.task_metrics[task_id] = metrics
            await self._emit("on_task_fail", task, last_exc)

    # -- Main entry point -----------------------------------------------------

    async def run(self) -> SchedulerMetrics:
        """Validate the task graph, then execute all tasks respecting
        dependency order, priority, and concurrency limits.

        Returns:
            A :class:`SchedulerMetrics` instance with timing data for every
            task.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task.
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        if not self._tasks:
            return self._metrics

        self._validate_dependencies()
        self._detect_cycles()

        levels = self._topological_sort()
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics = SchedulerMetrics()

        overall_start = time.monotonic()

        for level in levels:
            # Run all tasks in this level concurrently (bounded by semaphore)
            await asyncio.gather(
                *(self._execute_task(tid) for tid in level)
            )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # -- Convenience helpers --------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` for *task_id*.

        Raises:
            TaskNotFoundError: If no task with that ID exists.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task with id '{task_id}'") from None

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of all registered tasks."""
        return dict(self._tasks)

    @property
    def metrics(self) -> SchedulerMetrics:
        """Return the metrics from the most recent :meth:`run` call."""
        return self._metrics
