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
from typing import Any, Callable, Dict, List, Optional, Set


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
    """Represents a single schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority from 1 (lowest) to 10 (highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of times the task has been retried.
        max_retries: Maximum allowed retries before the task is marked FAILED.
        created_at: Timestamp when the task was created.
        result: The value returned by the task coroutine upon success.
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
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Per-task execution metrics."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds taken for the task."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a scheduler run."""

    total_start: Optional[float] = None
    total_end: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_time(self) -> Optional[float]:
        if self.total_start is not None and self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

# Type alias for coroutine-based task functions
TaskCoroutine = Callable[["Task"], Any]
EventCallback = Callable[["Task"], None]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control,
    retry logic, and an observer pattern for lifecycle events.

    Args:
        max_concurrency: Maximum number of tasks that may execute in parallel.
            Defaults to 4.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, TaskCoroutine] = {}
        self._max_concurrency = max_concurrency
        self._metrics = SchedulerMetrics()

        # Observer callbacks
        self._on_task_start: List[EventCallback] = []
        self._on_task_complete: List[EventCallback] = []
        self._on_task_fail: List[EventCallback] = []

    # ------------------------------------------------------------------
    # Public API – task registration
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coroutine: TaskCoroutine) -> None:
        """Register a task and its associated async coroutine.

        Args:
            task: Task metadata object.
            coroutine: An async callable that accepts a :class:`Task` and
                returns the task result.

        Raises:
            ValueError: If a task with the same ID has already been added.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine
        self._metrics.per_task[task.id] = TaskMetrics(task_id=task.id)

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def on_task_start(self, callback: EventCallback) -> None:
        """Register a callback invoked when a task transitions to RUNNING."""
        self._on_task_start.append(callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Register a callback invoked when a task transitions to COMPLETED."""
        self._on_task_complete.append(callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Register a callback invoked when a task transitions to FAILED."""
        self._on_task_fail.append(callback)

    def _emit(self, callbacks: List[EventCallback], task: Task) -> None:
        for cb in callbacks:
            try:
                cb(task)
            except Exception:
                pass  # Observers must not crash the scheduler

    # ------------------------------------------------------------------
    # Dependency / topology helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Check that all dependency IDs exist and no circular deps are present.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task ID.
            CircularDependencyError: If a cycle is detected in the task graph.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )
        self._detect_cycles()

    def _detect_cycles(self) -> None:
        """Kahn's algorithm topological sort to detect cycles."""
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks."
            )

    def _build_ready_queue(self, completed: Set[str]) -> List[Task]:
        """Return tasks whose dependencies are all satisfied, sorted by priority desc."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed for dep in task.dependencies):
                ready.append(task)
        # Higher priority value → run first
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task under the semaphore, with retry logic."""
        async with semaphore:
            metrics = self._metrics.per_task[task.id]
            while True:
                task.status = TaskStatus.RUNNING
                metrics.start_time = time.monotonic()
                self._emit(self._on_task_start, task)
                try:
                    result = await self._coroutines[task.id](task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    self._emit(self._on_task_complete, task)
                    return
                except Exception:
                    task.retry_count += 1
                    metrics.retry_count = task.retry_count
                    if task.retry_count <= task.max_retries:
                        # Exponential back-off: 0.1s * 2^(retry-1)
                        await asyncio.sleep(0.1 * (2 ** (task.retry_count - 1)))
                        task.status = TaskStatus.PENDING
                        continue
                    task.status = TaskStatus.FAILED
                    metrics.end_time = time.monotonic()
                    self._emit(self._on_task_fail, task)
                    return

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A :class:`SchedulerMetrics` instance with timing information for
            the entire run and each individual task.

        Raises:
            TaskNotFoundError: If any dependency references an unknown task.
            CircularDependencyError: If a dependency cycle exists.
        """
        self._validate_dependencies()

        self._metrics.total_start = time.monotonic()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        completed: Set[str] = set()
        running: Dict[str, asyncio.Task] = {}  # task_id -> asyncio.Task

        pending_ids = set(self._tasks.keys())

        while pending_ids or running:
            # Launch all tasks whose deps are now satisfied
            ready = self._build_ready_queue(completed)
            for task in ready:
                if task.id not in running:
                    coro = self._run_task(task, semaphore)
                    asyncio_task = asyncio.create_task(coro)
                    running[task.id] = asyncio_task
                    pending_ids.discard(task.id)

            if not running:
                # Nothing left to run and no pending tasks resolved → stuck
                break

            # Wait for at least one task to finish
            done, _ = await asyncio.wait(
                running.values(), return_when=asyncio.FIRST_COMPLETED
            )

            for asyncio_task in done:
                # Identify which task_id finished
                finished_id = next(
                    tid for tid, at in running.items() if at is asyncio_task
                )
                running.pop(finished_id)
                finished_task = self._tasks[finished_id]
                if finished_task.status == TaskStatus.COMPLETED:
                    completed.add(finished_id)
                # FAILED tasks are left out of `completed`; dependents will stall
                # and eventually be collected by the outer loop exit.

        self._metrics.total_end = time.monotonic()
        return self._metrics

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return a registered task by ID.

        Raises:
            TaskNotFoundError: If no task with the given ID exists.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task found with id '{task_id}'.")

    def get_metrics(self) -> SchedulerMetrics:
        """Return the aggregate metrics object for the last (or current) run."""
        return self._metrics

    def get_all_tasks(self) -> List[Task]:
        """Return all registered tasks in insertion order."""
        return list(self._tasks.values())
