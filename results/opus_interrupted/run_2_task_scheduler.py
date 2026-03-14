"""
Async task scheduler with priority queue, dependency resolution,
retry logic, concurrency control, and observer-pattern event emission.
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
# Enums & Exceptions
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when the dependency graph contains a cycle."""


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a unit of work managed by the scheduler.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs this task depends on.
        status: Current lifecycle status.
        retry_count: How many times the task has been retried so far.
        max_retries: Maximum allowed retries before permanent failure.
        created_at: Timestamp of task creation.
        result: Outcome value after execution (success or exception).
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
class ExecutionMetrics:
    """Aggregate metrics collected during a scheduler run.

    Attributes:
        total_time: Wall-clock seconds for the entire run.
        per_task_time: Mapping of task ID -> seconds spent executing.
        retry_counts: Mapping of task ID -> number of retries used.
    """

    total_time: float = 0.0
    per_task_time: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

# Type alias for the async callable that actually performs a task's work.
TaskExecutor = Callable[[Task], Coroutine[Any, Any, Any]]

# Type alias for event listener callbacks.
EventListener = Callable[[Task], Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control,
    retry logic, and an observer-pattern event system.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
        base_backoff: Base delay (seconds) for exponential-backoff retries.
    """

    # Event names
    EVENT_START = "on_task_start"
    EVENT_COMPLETE = "on_task_complete"
    EVENT_FAIL = "on_task_fail"

    def __init__(
        self,
        max_concurrency: int = 4,
        base_backoff: float = 1.0,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.base_backoff = base_backoff

        self._tasks: dict[str, Task] = {}
        self._executors: dict[str, TaskExecutor] = {}
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self._metrics = ExecutionMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(self, task: Task, executor: TaskExecutor) -> None:
        """Register a task and its async executor function.

        Args:
            task: The task to schedule.
            executor: An async callable that performs the task's work.
                      It receives the ``Task`` instance and should return
                      a result value.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._executors[task.id] = executor

    # -- Observer pattern ----------------------------------------------------

    def on(self, event: str, listener: EventListener) -> None:
        """Subscribe *listener* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.
        """
        self._listeners[event].append(listener)

    def _emit(self, event: str, task: Task) -> None:
        """Fire all listeners registered for *event*."""
        for listener in self._listeners.get(event, []):
            listener(task)

    # -- Dependency graph helpers --------------------------------------------

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from registered tasks.

        Returns:
            (adjacency, in_degree) where *adjacency[dep]* lists tasks that
            depend on *dep*, and *in_degree[task_id]* is the number of
            unsatisfied dependencies.
        """
        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    raise ValueError(
                        f"Task '{tid}' depends on unknown task '{dep}'"
                    )
                adjacency[dep].append(tid)
                in_degree[tid] += 1

        return adjacency, in_degree

    def _detect_circular_dependencies(self) -> None:
        """Raise ``CircularDependencyError`` if the graph has a cycle."""
        adjacency, in_degree = self._build_graph()
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks"
            )

    def get_execution_plan(self) -> list[list[str]]:
        """Return an ordered list of execution groups (topological layers).

        Within each group the tasks are independent and may run concurrently.
        Groups are sorted so that higher-priority tasks appear first within
        each layer.

        Returns:
            A list of groups, where each group is a list of task IDs.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._detect_circular_dependencies()

        adjacency, in_degree = self._build_graph()
        groups: list[list[str]] = []

        ready = [tid for tid, deg in in_degree.items() if deg == 0]

        while ready:
            # Sort by descending priority so higher-priority tasks come first.
            ready.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(list(ready))

            next_ready: list[str] = []
            for tid in ready:
                for neighbor in adjacency[tid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = next_ready

        return groups

    # -- Execution -----------------------------------------------------------

    async def _run_task(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task with retry + exponential backoff.

        The semaphore gates concurrency across all tasks.
        """
        async with semaphore:
            while True:
                task.status = TaskStatus.RUNNING
                self._emit(self.EVENT_START, task)
                start = time.monotonic()

                try:
                    task.result = await self._executors[task.id](task)
                    elapsed = time.monotonic() - start
                    task.status = TaskStatus.COMPLETED
                    self._metrics.per_task_time[task.id] = elapsed
                    self._metrics.retry_counts[task.id] = task.retry_count
                    self._emit(self.EVENT_COMPLETE, task)
                    return

                except Exception as exc:
                    elapsed = time.monotonic() - start
                    task.retry_count += 1

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.result = exc
                        self._metrics.per_task_time[task.id] = elapsed
                        self._metrics.retry_counts[task.id] = task.retry_count
                        self._emit(self.EVENT_FAIL, task)
                        return

                    delay = self.base_backoff * (2 ** (task.retry_count - 1))
                    await asyncio.sleep(delay)

    async def run(self) -> ExecutionMetrics:
        """Execute all registered tasks respecting dependencies, concurrency,
        and retry policies.

        Returns:
            ``ExecutionMetrics`` with timing and retry statistics.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        plan = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        self._metrics = ExecutionMetrics()
        overall_start = time.monotonic()

        for group in plan:
            tasks_in_group = [self._tasks[tid] for tid in group]
            await asyncio.gather(
                *(self._run_task(t, semaphore) for t in tasks_in_group)
            )

            # If any dependency in the group failed, mark downstream tasks as
            # failed too (they will never be reached because they stay in
            # later groups, but we set their status proactively).
            failed_ids = {t.id for t in tasks_in_group if t.status == TaskStatus.FAILED}
            if failed_ids:
                for future_group in plan:
                    for tid in future_group:
                        t = self._tasks[tid]
                        if t.status == TaskStatus.PENDING and failed_ids & set(t.dependencies):
                            t.status = TaskStatus.FAILED
                            t.result = RuntimeError(
                                f"Dependency failed: {failed_ids & set(t.dependencies)}"
                            )
                            self._emit(self.EVENT_FAIL, t)

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # -- Introspection -------------------------------------------------------

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of registered tasks."""
        return dict(self._tasks)

    @property
    def metrics(self) -> ExecutionMetrics:
        """Most recent execution metrics."""
        return self._metrics
