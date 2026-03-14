"""
Async task scheduler with priority queuing, dependency resolution,
retry logic, concurrency control, and an observer-based event system.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        if cycle:
            msg = f"Circular dependency detected: {' -> '.join(cycle)}"
        else:
            msg = "Circular dependency detected in the task graph"
        super().__init__(msg)
        self.cycle = cycle


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work inside the scheduler.

    Attributes:
        id:           Unique identifier for the task.
        name:         Human-readable task name.
        priority:     Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task.
        status:       Current lifecycle status.
        retry_count:  How many times the task has been retried so far.
        max_retries:  Maximum number of retry attempts before final failure.
        created_at:   Timestamp of task creation.
        result:       The return value of the task coroutine (set after completion).
    """
    id: str
    name: str
    priority: int = 5
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Timing and retry information for a single task execution."""
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    retries: int = 0

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds the task took (including retries)."""
        return self.end_time - self.start_time


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for a full scheduler run."""
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

    Features:
        * Priority-aware execution within each dependency tier.
        * Topological-sort-based dependency resolution.
        * Circular-dependency detection.
        * Configurable concurrency limit.
        * Exponential-backoff retry on failure.
        * Observer-pattern event hooks (on_task_start, on_task_complete,
          on_task_fail).

    Example::

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(Task(id="a", name="Step A"), coro=my_coroutine)
        plan = scheduler.get_execution_plan()
        metrics = await scheduler.run()
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max_concurrency = max_concurrency
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._metrics = SchedulerMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The :class:`Task` descriptor.
            coro: An async callable that will be awaited when the task runs.
                  It receives the :class:`Task` instance as its sole argument.

        Raises:
            ValueError: If a task with the same *id* is already registered.
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
        """Fire all callbacks registered for *event*."""
        for cb in self._listeners.get(event, []):
            ret = cb(*args, **kwargs)
            if asyncio.iscoroutine(ret):
                await ret

    # -- Dependency graph helpers ---------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency references a registered task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_circular_dependencies(self) -> None:
        """Raise :class:`CircularDependencyError` if a cycle exists.

        Uses iterative DFS with explicit *visiting* / *visited* sets to
        detect back-edges and reconstruct the cycle path.
        """
        visited: set[str] = set()
        visiting: set[str] = set()

        def _dfs(task_id: str, path: list[str]) -> None:
            if task_id in visiting:
                # Extract the cycle from path
                cycle_start = path.index(task_id)
                cycle = path[cycle_start:] + [task_id]
                raise CircularDependencyError(cycle)
            if task_id in visited:
                return
            visiting.add(task_id)
            path.append(task_id)
            for dep_id in self._tasks[task_id].dependencies:
                _dfs(dep_id, path)
            path.pop()
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self._tasks:
            if task_id not in visited:
                _dfs(task_id, [])

    def _topological_sort(self) -> list[list[str]]:
        """Return tasks grouped into dependency tiers (Kahn's algorithm).

        Each tier is a list of task IDs whose dependencies are fully
        satisfied by earlier tiers.  Within each tier the tasks are
        sorted by descending priority so higher-priority work starts
        first.

        Returns:
            A list of tiers, where each tier is a list of task IDs.

        Raises:
            CircularDependencyError: If the graph contains a cycle (safety
                net – callers should run ``_detect_circular_dependencies``
                first for a better error message).
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for tid, task in self._tasks.items():
            for dep_id in task.dependencies:
                in_degree[tid] += 1
                dependents[dep_id].append(tid)

        # Seed with zero-in-degree nodes
        queue: list[str] = [tid for tid, deg in in_degree.items() if deg == 0]
        tiers: list[list[str]] = []
        processed = 0

        while queue:
            # Sort current tier by priority descending
            queue.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            tiers.append(list(queue))
            processed += len(queue)
            next_queue: list[str] = []
            for tid in queue:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            queue = next_queue

        if processed != len(self._tasks):
            raise CircularDependencyError()

        return tiers

    # -- Execution plan -------------------------------------------------------

    def get_execution_plan(self) -> list[list[str]]:
        """Return the tiered execution plan without running anything.

        Validates dependencies and checks for cycles first.

        Returns:
            A list of tiers (each tier is a list of task IDs).
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()
        return self._topological_sort()

    # -- Single-task execution with retries -----------------------------------

    async def _execute_task(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task with exponential-backoff retry.

        Args:
            task: The task to run.
            semaphore: Concurrency-limiting semaphore.
        """
        metrics = TaskMetrics(task_id=task.id)
        metrics.start_time = time.monotonic()

        async with semaphore:
            while True:
                try:
                    task.status = TaskStatus.RUNNING
                    await self._emit("on_task_start", task)
                    result = await self._coroutines[task.id](task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retries = task.retry_count
                    self._metrics.task_metrics[task.id] = metrics
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        metrics.end_time = time.monotonic()
                        metrics.retries = task.retry_count - 1
                        self._metrics.task_metrics[task.id] = metrics
                        await self._emit("on_task_fail", task, exc)
                        return
                    # Exponential backoff: 2^(retry - 1) * 0.1 seconds
                    backoff = (2 ** (task.retry_count - 1)) * 0.1
                    await asyncio.sleep(backoff)

    # -- Full scheduler run ---------------------------------------------------

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            :class:`SchedulerMetrics` with timing information for every task.

        Raises:
            ValueError: If a dependency references an unknown task.
            CircularDependencyError: If the task graph contains a cycle.
        """
        tiers = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()

        for tier in tiers:
            # Check that all dependencies in this tier completed successfully
            tasks_to_run: list[Task] = []
            for tid in tier:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[d].status == TaskStatus.COMPLETED
                    for d in task.dependencies
                )
                if deps_ok:
                    tasks_to_run.append(task)
                else:
                    task.status = TaskStatus.FAILED
                    await self._emit(
                        "on_task_fail",
                        task,
                        RuntimeError("Dependency failed"),
                    )

            # Run all eligible tasks in this tier concurrently
            if tasks_to_run:
                await asyncio.gather(
                    *(self._execute_task(t, semaphore) for t in tasks_to_run)
                )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # -- Convenience helpers --------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` with the given *task_id*.

        Raises:
            KeyError: If no such task is registered.
        """
        return self._tasks[task_id]

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of registered tasks."""
        return dict(self._tasks)

    def reset(self) -> None:
        """Reset all tasks to PENDING and clear metrics."""
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.result = None
        self._metrics = SchedulerMetrics()
