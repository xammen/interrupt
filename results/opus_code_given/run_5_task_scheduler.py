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
    """Represents a single unit of work in the scheduler.

    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: IDs of tasks that must complete before this one.
        status: Current execution status.
        retry_count: How many times execution has been retried.
        max_retries: Maximum allowed retries before marking as FAILED.
        created_at: Timestamp of task creation.
        result: Return value of the task coroutine (set after completion).
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
class ExecutionMetrics:
    """Aggregate metrics collected during a scheduler run.

    Attributes:
        total_time: Wall-clock seconds for the entire run.
        per_task_time: Mapping of task ID to its wall-clock seconds.
        retry_counts: Mapping of task ID to number of retries performed.
    """

    total_time: float = 0.0
    per_task_time: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

# Type alias for the async callable that a task executes.
TaskCoroutine = Callable[[Task], Coroutine[Any, Any, Any]]

# Type alias for event listeners.
EventListener = Callable[[Task], Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
        base_backoff: Base delay (seconds) for exponential-backoff retries.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        base_backoff: float = 1.0,
    ) -> None:
        self._tasks: dict[str, Task] = {}
        self._task_coroutines: dict[str, TaskCoroutine] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self.metrics = ExecutionMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(self, task: Task, coro: TaskCoroutine) -> None:
        """Register a task and its associated coroutine.

        Args:
            task: The Task instance to schedule.
            coro: An async callable ``coro(task)`` that performs the work.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._task_coroutines[task.id] = coro

    # -- Observer pattern -----------------------------------------------------

    def on(self, event: str, listener: EventListener) -> None:
        """Subscribe *listener* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.
        """
        self._listeners[event].append(listener)

    def _emit(self, event: str, task: Task) -> None:
        """Notify all listeners registered for *event*."""
        for listener in self._listeners.get(event, []):
            listener(task)

    # -- Dependency graph utilities -------------------------------------------

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from registered tasks.

        Returns:
            A tuple of (adjacency list, in-degree map).
        """
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    raise ValueError(
                        f"Task '{tid}' depends on unknown task '{dep}'"
                    )
                adj[dep].append(tid)
                in_degree[tid] += 1

        return adj, in_degree

    def _topological_sort(self) -> list[list[str]]:
        """Kahn's algorithm producing *groups* of concurrently-runnable tasks.

        Each group contains tasks whose dependencies are fully satisfied by
        the preceding groups.  Within a group, tasks are sorted by descending
        priority so the highest-priority tasks are picked first when the
        concurrency semaphore is limited.

        Returns:
            Ordered list of task-ID groups.

        Raises:
            CircularDependencyError: If a cycle exists in the dependency graph.
        """
        adj, in_degree = self._build_graph()
        queue: list[str] = [tid for tid, deg in in_degree.items() if deg == 0]
        groups: list[list[str]] = []
        visited = 0

        while queue:
            # Sort current frontier by priority (highest first).
            queue.sort(key=lambda t: self._tasks[t].priority, reverse=True)
            groups.append(list(queue))
            visited += len(queue)
            next_queue: list[str] = []
            for tid in queue:
                for neighbour in adj[tid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        next_queue.append(neighbour)
            queue = next_queue

        if visited != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks"
            )

        return groups

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        The outer list is ordered: group *i* must finish before group *i+1*
        starts.

        Raises:
            CircularDependencyError: If a cycle exists.
        """
        return self._topological_sort()

    # -- Execution ------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + exponential backoff.

        The semaphore limits how many tasks run concurrently.
        """
        async with semaphore:
            coro = self._task_coroutines[task.id]

            while True:
                task.status = TaskStatus.RUNNING
                self._emit("on_task_start", task)
                start = time.monotonic()

                try:
                    task.result = await coro(task)
                    elapsed = time.monotonic() - start
                    task.status = TaskStatus.COMPLETED
                    self.metrics.per_task_time[task.id] = elapsed
                    self.metrics.retry_counts[task.id] = task.retry_count
                    self._emit("on_task_complete", task)
                    return
                except Exception:
                    elapsed = time.monotonic() - start
                    task.retry_count += 1

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        self.metrics.per_task_time[task.id] = elapsed
                        self.metrics.retry_counts[task.id] = task.retry_count
                        self._emit("on_task_fail", task)
                        return

                    # Exponential backoff: base * 2^(attempt-1)
                    delay = self._base_backoff * (2 ** (task.retry_count - 1))
                    await asyncio.sleep(delay)

    async def run(self) -> ExecutionMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            ExecutionMetrics with timing and retry information.

        Raises:
            CircularDependencyError: If a cycle exists.
        """
        self.metrics = ExecutionMetrics()
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()

        for group in groups:
            tasks_in_group = [self._tasks[tid] for tid in group]
            await asyncio.gather(
                *(self._run_task(t, semaphore) for t in tasks_in_group)
            )
            # If any dependency failed, mark dependents as failed too.
            failed_ids = {
                t.id for t in tasks_in_group if t.status == TaskStatus.FAILED
            }
            if failed_ids:
                self._cascade_failure(failed_ids)

        self.metrics.total_time = time.monotonic() - overall_start
        return self.metrics

    def _cascade_failure(self, failed_ids: set[str]) -> None:
        """Mark all transitive dependents of *failed_ids* as FAILED."""
        adj, _ = self._build_graph()
        visited: set[str] = set()
        stack = list(failed_ids)

        while stack:
            tid = stack.pop()
            for dep in adj.get(tid, []):
                if dep not in visited and self._tasks[dep].status == TaskStatus.PENDING:
                    self._tasks[dep].status = TaskStatus.FAILED
                    self._emit("on_task_fail", self._tasks[dep])
                    visited.add(dep)
                    stack.append(dep)
