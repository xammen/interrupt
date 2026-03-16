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

    def _detect_circular_dependencies(self) -> None:
        """Raise :class:`CircularDependencyError` if the graph has cycles.

        Uses iterative DFS with explicit *WHITE / GRAY / BLACK* colouring.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        for start in self._tasks:
            if colour[start] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx == 0:
                    colour[node] = GRAY
                if idx < len(deps):
                    stack.append((node, idx + 1))
                    dep = deps[idx]
                    if colour[dep] == GRAY:
                        # Reconstruct cycle path.
                        cycle = [dep, node]
                        cur = parent.get(node)
                        while cur is not None and cur != dep:
                            cycle.append(cur)
                            cur = parent.get(cur)
                        cycle.append(dep)
                        cycle.reverse()
                        raise CircularDependencyError(cycle)
                    if colour[dep] == WHITE:
                        parent[dep] = node
                        stack.append((dep, 0))
                else:
                    colour[node] = BLACK

    def _topological_groups(self) -> list[list[str]]:
        """Return tasks grouped by topological level (Kahn's algorithm).

        Tasks within the same group are independent and can run concurrently.
        Within each group tasks are sorted by descending priority so that
        higher-priority tasks start first.

        Returns:
            Ordered list of groups, where each group is a list of task IDs.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        groups: list[list[str]] = []

        while queue:
            # Sort current level by descending priority.
            queue.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(list(queue))
            next_queue: list[str] = []
            for tid in queue:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            queue = next_queue

        return groups

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that may execute concurrently.
        The outer list is in dependency order.

        Raises:
            CircularDependencyError: If the task graph contains a cycle.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()
        return self._topological_groups()

    # -- Execution engine -----------------------------------------------------

    async def _run_task(self, task_id: str) -> None:
        """Execute a single task with retry + exponential backoff.

        The semaphore is acquired *outside* this method by the caller so that
        the concurrency limit is enforced.
        """
        task = self._tasks[task_id]
        metrics = TaskMetrics(task_id=task_id)
        self._metrics.task_metrics[task_id] = metrics

        while True:
            task.status = TaskStatus.RUNNING
            await self._emit("on_task_start", task)
            metrics.start_time = time.monotonic()

            try:
                result = await self._coroutines[task_id]()
                metrics.end_time = time.monotonic()
                task.status = TaskStatus.COMPLETED
                task.result = result
                await self._emit("on_task_complete", task)
                return
            except Exception as exc:
                metrics.end_time = time.monotonic()
                task.retry_count += 1
                metrics.retries = task.retry_count

                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    await self._emit("on_task_fail", task, exc)
                    return

                # Exponential backoff: 2^(retry-1) seconds (0.5, 1, 2, …)
                backoff = 2 ** (task.retry_count - 1) * 0.5
                await asyncio.sleep(backoff)

    async def _run_task_with_semaphore(self, task_id: str) -> None:
        """Wrap :meth:`_run_task` with the concurrency-limiting semaphore."""
        assert self._semaphore is not None
        async with self._semaphore:
            await self._run_task(task_id)

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A :class:`SchedulerMetrics` instance with timing data.

        Raises:
            CircularDependencyError: If the task graph contains a cycle.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()
        groups = self._topological_groups()

        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics = SchedulerMetrics()

        total_start = time.monotonic()

        for group in groups:
            # Filter out tasks whose dependencies failed.
            runnable: list[str] = []
            for tid in group:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[d].status == TaskStatus.COMPLETED
                    for d in task.dependencies
                )
                if deps_ok:
                    runnable.append(tid)
                else:
                    task.status = TaskStatus.FAILED
                    self._metrics.task_metrics[tid] = TaskMetrics(task_id=tid)

            await asyncio.gather(
                *(self._run_task_with_semaphore(tid) for tid in runnable)
            )

        self._metrics.total_time = time.monotonic() - total_start
        return self._metrics
