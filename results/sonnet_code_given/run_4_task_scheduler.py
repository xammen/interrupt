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
        """Ensure every dependency ID references a registered task.

        Raises:
            TaskNotFoundError: If an unknown dependency ID is encountered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Perform a Kahn's-algorithm topological sort and group tasks into
        independent execution layers.

        Returns:
            A list of groups where each group is a list of task IDs whose
            dependencies were all satisfied by previous groups.

        Raises:
            CircularDependencyError: If a cycle is detected.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Seed the queue with tasks that have no dependencies,
        # ordered by descending priority so higher-priority tasks
        # appear first within the same layer.
        queue: deque[str] = deque(
            sorted(
                [tid for tid, deg in in_degree.items() if deg == 0],
                key=lambda tid: -self._tasks[tid].priority,
            )
        )

        groups: List[List[str]] = []
        visited = 0

        while queue:
            # Drain the current layer
            layer_size = len(queue)
            layer: List[str] = []
            for _ in range(layer_size):
                tid = queue.popleft()
                layer.append(tid)
                visited += 1
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            # Re-sort newly added items by priority
            queue = deque(sorted(queue, key=lambda tid: -self._tasks[tid].priority))
            groups.append(layer)

        if visited != len(self._tasks):
            # Some nodes were never reached — cycle detected
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Execution plan
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can run concurrently.
        Groups are ordered such that all dependencies of a group are
        satisfied by the preceding groups.

        Returns:
            List of task-ID groups in execution order.

        Raises:
            CircularDependencyError: If a cycle is detected.
            TaskNotFoundError: If an unknown dependency is referenced.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting priorities, dependencies,
        concurrency limits, and retry logic.

        Returns:
            :class:`SchedulerMetrics` populated with timing data.

        Raises:
            CircularDependencyError: If a cycle is detected.
            TaskNotFoundError: If an unknown dependency is referenced.
        """
        self._validate_dependencies()
        groups = self._topological_sort()

        self._metrics.total_start = time.monotonic()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in groups:
            aws = [self._run_task(tid, semaphore) for tid in group]
            await asyncio.gather(*aws)

        self._metrics.total_end = time.monotonic()
        return self._metrics

    async def _run_task(self, task_id: str, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry / exponential-backoff logic.

        Args:
            task_id: ID of the task to execute.
            semaphore: Shared concurrency limiter.
        """
        task = self._tasks[task_id]
        coroutine = self._coroutines[task_id]
        metrics = self._metrics.per_task[task_id]

        async with semaphore:
            task.status = TaskStatus.RUNNING
            metrics.start_time = time.monotonic()
            self._emit(self._on_task_start, task)

            attempt = 0
            while True:
                try:
                    task.result = await coroutine(task)
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    self._emit(self._on_task_complete, task)
                    return
                except Exception:
                    attempt += 1
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        metrics.end_time = time.monotonic()
                        metrics.retry_count = task.retry_count
                        self._emit(self._on_task_fail, task)
                        return
                    # Exponential backoff: 2^(attempt-1) seconds, capped at 30 s
                    backoff = min(2 ** (attempt - 1), 30)
                    await asyncio.sleep(backoff)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> SchedulerMetrics:
        """Return the current scheduler metrics snapshot."""
        return self._metrics

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` for the given ID.

        Raises:
            TaskNotFoundError: If the ID is not registered.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task registered with id '{task_id}'.")

    def completed_tasks(self) -> List[Task]:
        """Return all tasks that finished with COMPLETED status."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]

    def failed_tasks(self) -> List[Task]:
        """Return all tasks that finished with FAILED status."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]
