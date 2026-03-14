"""
task_scheduler.py - Async task scheduler with priority queue, dependency resolution,
concurrent execution, retry logic, and observer pattern.
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
# Exceptions
# ---------------------------------------------------------------------------

class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist."""


# ---------------------------------------------------------------------------
# Enums / Dataclasses
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work managed by the scheduler.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status of the task.
        retry_count: Number of times the task has been retried.
        max_retries: Maximum number of retry attempts allowed.
        created_at: Timestamp when the task was created.
        result: Stores the return value (or exception) after execution.
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
        if not (1 <= self.priority <= 10):
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


@dataclass
class TaskMetrics:
    """Execution metrics collected for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds between start and end."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EventCallback = Callable[[Task], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrent execution.

    Features
    --------
    - Priority queue: higher-priority tasks run first within each execution group.
    - Dependency resolution via topological sort (Kahn's algorithm).
    - Circular dependency detection with :class:`CircularDependencyError`.
    - Concurrent execution of independent tasks bounded by *max_concurrency*.
    - Exponential backoff retry logic for failed tasks.
    - Observer pattern via ``on_task_start``, ``on_task_complete``, ``on_task_fail``.
    - :meth:`get_execution_plan` returns ordered groups of task IDs.
    - Per-task and total execution metrics.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks that may run in parallel (default: 4).
    base_backoff:
        Base delay in seconds for exponential backoff (default: 1.0).
    """

    def __init__(self, max_concurrency: int = 4, base_backoff: float = 1.0) -> None:
        self._tasks: Dict[str, Task] = {}
        self._callables: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff

        # Observer callbacks
        self._on_start_handlers: List[EventCallback] = []
        self._on_complete_handlers: List[EventCallback] = []
        self._on_fail_handlers: List[EventCallback] = []

        # Metrics
        self._metrics: Dict[str, TaskMetrics] = {}
        self._total_start: Optional[float] = None
        self._total_end: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API – task registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro_fn: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Parameters
        ----------
        task:
            Task descriptor.
        coro_fn:
            An *async* callable that performs the actual work. It is invoked
            with no arguments; close over any required context before
            registering.
        """
        self._tasks[task.id] = task
        self._callables[task.id] = coro_fn
        self._metrics[task.id] = TaskMetrics(task_id=task.id)

    # ------------------------------------------------------------------
    # Public API – observers
    # ------------------------------------------------------------------

    def on_task_start(self, handler: EventCallback) -> None:
        """Register a coroutine callback invoked when a task begins execution."""
        self._on_start_handlers.append(handler)

    def on_task_complete(self, handler: EventCallback) -> None:
        """Register a coroutine callback invoked when a task succeeds."""
        self._on_complete_handlers.append(handler)

    def on_task_fail(self, handler: EventCallback) -> None:
        """Register a coroutine callback invoked when a task fails all retries."""
        self._on_fail_handlers.append(handler)

    # ------------------------------------------------------------------
    # Public API – execution plan
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the topologically sorted execution groups.

        Each inner list contains task IDs that can run concurrently
        (all their dependencies are satisfied by earlier groups).

        Returns
        -------
        List[List[str]]
            Ordered groups; tasks within a group are sorted by priority
            descending (highest first).

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        TaskNotFoundError
            If a dependency references an unknown task ID.
        """
        return self._build_execution_groups()

    # ------------------------------------------------------------------
    # Public API – run
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, TaskMetrics]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns
        -------
        Dict[str, TaskMetrics]
            Per-task metrics keyed by task ID, plus the key ``"__total__"``
            containing aggregate timing.
        """
        groups = self._build_execution_groups()
        self._total_start = time.monotonic()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in groups:
            # Sort group by priority descending so higher-priority tasks
            # are submitted to the event loop first.
            sorted_group = sorted(
                group, key=lambda tid: self._tasks[tid].priority, reverse=True
            )
            await asyncio.gather(
                *(self._run_task_with_retry(tid, semaphore) for tid in sorted_group)
            )

        self._total_end = time.monotonic()
        total = TaskMetrics(task_id="__total__")
        total.start_time = self._total_start
        total.end_time = self._total_end
        return {**self._metrics, "__total__": total}

    # ------------------------------------------------------------------
    # Internal – dependency / topological sort
    # ------------------------------------------------------------------

    def _build_execution_groups(self) -> List[List[str]]:
        """Perform Kahn's topological sort and return execution groups."""
        # Validate all dependency references
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

        # Build adjacency structures
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)  # dep -> [tasks that need dep]

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Kahn's BFS
        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: List[List[str]] = []
        visited: Set[str] = set()

        while queue:
            # Drain the current frontier into one execution group
            current_group = list(queue)
            queue.clear()
            groups.append(current_group)
            visited.update(current_group)

            next_frontier: List[str] = []
            for tid in current_group:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_frontier.append(dependent)
            queue.extend(next_frontier)

        if len(visited) != len(self._tasks):
            cycle_nodes = set(self._tasks) - visited
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Internal – task execution with retry
    # ------------------------------------------------------------------

    async def _run_task_with_retry(
        self, task_id: str, semaphore: asyncio.Semaphore
    ) -> None:
        """Run a single task, retrying with exponential backoff on failure."""
        task = self._tasks[task_id]
        metrics = self._metrics[task_id]

        async with semaphore:
            task.status = TaskStatus.RUNNING
            metrics.start_time = time.monotonic()
            await self._emit(self._on_start_handlers, task)

            attempt = 0
            while True:
                try:
                    coro_fn = self._callables[task_id]
                    result = await coro_fn()
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    await self._emit(self._on_complete_handlers, task)
                    return
                except Exception as exc:  # noqa: BLE001
                    attempt += 1
                    task.retry_count += 1
                    if task.retry_count <= task.max_retries:
                        backoff = self._base_backoff * (2 ** (task.retry_count - 1))
                        await asyncio.sleep(backoff)
                    else:
                        task.result = exc
                        task.status = TaskStatus.FAILED
                        metrics.end_time = time.monotonic()
                        metrics.retry_count = task.retry_count
                        await self._emit(self._on_fail_handlers, task)
                        return

    # ------------------------------------------------------------------
    # Internal – event emission
    # ------------------------------------------------------------------

    @staticmethod
    async def _emit(handlers: List[EventCallback], task: Task) -> None:
        """Invoke all registered handlers for an event."""
        for handler in handlers:
            await handler(task)
