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
        """Build an adjacency list and in-degree map from registered tasks.

        Returns:
            A tuple of ``(adjacency, in_degree)`` where *adjacency* maps each
            task ID to the list of task IDs that depend on it, and *in_degree*
            maps each task ID to the number of unresolved dependencies.

        Raises:
            ValueError: If a task declares a dependency on an unknown task ID.
        """
        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{tid}' depends on unknown task '{dep_id}'"
                    )
                adjacency[dep_id].append(tid)
                in_degree[tid] += 1

        return dict(adjacency), in_degree

    def _detect_cycle(self) -> None:
        """Raise :class:`CircularDependencyError` if the graph has a cycle.

        Uses Kahn's algorithm: if a full topological ordering cannot be
        produced, a cycle must exist.
        """
        adjacency, in_degree = self._build_graph()
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbour in adjacency.get(node, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if visited != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks"
            )

    # -- Single-task execution ------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + exponential backoff.

        On success the task's *status* is set to ``COMPLETED`` and *result*
        holds the coroutine's return value.  On final failure *status* is set
        to ``FAILED``.

        Args:
            task: The task to execute.
            semaphore: Semaphore used to enforce concurrency limits.
        """
        coro = self._task_coroutines[task.id]

        async with semaphore:
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

    # -- Full execution -------------------------------------------------------

    async def execute_all(self) -> ExecutionMetrics:
        """Run all registered tasks respecting dependencies and concurrency.

        Tasks are dispatched using a topological ordering.  Among tasks whose
        dependencies are satisfied, higher-priority tasks are started first.

        Returns:
            An :class:`ExecutionMetrics` instance with timing information.

        Raises:
            CircularDependencyError: If a dependency cycle is detected.
            ValueError: If a dependency references an unknown task.
        """
        if not self._tasks:
            return self.metrics

        self._detect_cycle()
        adjacency, in_degree = self._build_graph()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        # Track running asyncio.Tasks so we can await them.
        running: dict[str, asyncio.Task[None]] = {}
        # Collect completed task ids to know when dependents are ready.
        completed: set[str] = set()
        # Track failed task ids so dependents are not started.
        failed: set[str] = set()

        overall_start = time.monotonic()

        def _ready_tasks() -> list[Task]:
            """Return PENDING tasks whose dependencies are all completed."""
            ready = []
            for tid, task in self._tasks.items():
                if task.status != TaskStatus.PENDING:
                    continue
                if all(d in completed for d in task.dependencies):
                    ready.append(task)
            # Higher priority first.
            ready.sort(key=lambda t: t.priority, reverse=True)
            return ready

        while True:
            # Launch all ready tasks.
            for task in _ready_tasks():
                task_handle = asyncio.create_task(self._run_task(task, semaphore))
                running[task.id] = task_handle

            if not running:
                # Nothing running and nothing ready — we're done.
                break

            # Wait for at least one task to finish.
            done, _ = await asyncio.wait(
                running.values(), return_when=asyncio.FIRST_COMPLETED
            )

            # Process finished tasks.
            finished_ids = [
                tid for tid, handle in running.items() if handle in done
            ]
            for tid in finished_ids:
                del running[tid]
                task = self._tasks[tid]
                if task.status == TaskStatus.COMPLETED:
                    completed.add(tid)
                else:
                    failed.add(tid)
                    # Mark all transitive dependents as FAILED.
                    self._fail_dependents(tid, adjacency, failed)

        self.metrics.total_time = time.monotonic() - overall_start
        return self.metrics

    def _fail_dependents(
        self,
        tid: str,
        adjacency: dict[str, list[str]],
        failed: set[str],
    ) -> None:
        """Recursively mark dependents of a failed task as FAILED."""
        for dep_id in adjacency.get(tid, []):
            if dep_id not in failed:
                task = self._tasks[dep_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.FAILED
                    self._emit("on_task_fail", task)
                    failed.add(dep_id)
                    self._fail_dependents(dep_id, adjacency, failed)

    # -- Accessors ------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[Task]:
        """Return the :class:`Task` with the given ID, or ``None``."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        """Return all registered tasks as a list."""
        return list(self._tasks.values())
