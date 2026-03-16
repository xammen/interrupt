"""
task_scheduler.py - Async task scheduler with priority queue, dependency resolution,
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
    """Raised when a referenced task ID does not exist."""


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
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of retries attempted so far.
        max_retries: Maximum number of retry attempts before marking FAILED.
        created_at: Timestamp of task creation.
        result: Stores the return value after successful execution.
        _fn: The async callable to execute (not included in repr by default).
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
    _fn: Optional[Callable[[], Coroutine[Any, Any, Any]]] = field(
        default=None, repr=False, compare=False
    )

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
    start_time: float = 0.0
    end_time: float = 0.0
    retry_count: int = 0

    @property
    def elapsed(self) -> float:
        """Wall-clock time in seconds from start to end."""
        return self.end_time - self.start_time


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for an entire scheduler run."""

    total_start: float = 0.0
    total_end: float = 0.0
    tasks: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> float:
        return self.total_end - self.total_start


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency limiting,
    exponential-backoff retries, and an observer pattern for lifecycle events.

    Args:
        max_concurrency: Maximum number of tasks that may run simultaneously.
                         Defaults to 4.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: Dict[str, Task] = {}
        self._max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.metrics = SchedulerMetrics()

        # Observer callbacks
        self._on_task_start: List[Callable[[Task], None]] = []
        self._on_task_complete: List[Callable[[Task], None]] = []
        self._on_task_fail: List[Callable[[Task, Exception], None]] = []

    # ------------------------------------------------------------------
    # Public API — task registration
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        """Register a task with the scheduler.

        Args:
            task: The :class:`Task` instance to register.

        Raises:
            ValueError: If a task with the same ID already exists.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already registered.")
        self._tasks[task.id] = task

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def on_task_start(self, callback: Callable[[Task], None]) -> None:
        """Register a callback invoked just before a task begins execution."""
        self._on_task_start.append(callback)

    def on_task_complete(self, callback: Callable[[Task], None]) -> None:
        """Register a callback invoked after a task completes successfully."""
        self._on_task_complete.append(callback)

    def on_task_fail(self, callback: Callable[[Task, Exception], None]) -> None:
        """Register a callback invoked after a task exhausts all retries."""
        self._on_task_fail.append(callback)

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency reference points to a registered task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return execution groups via Kahn's algorithm (topological sort).

        Each group contains task IDs that can run concurrently because all
        their dependencies are satisfied by previous groups.

        Returns:
            Ordered list of groups, e.g. [["a"], ["b", "c"], ["d"]].

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
        groups: List[List[str]] = []
        visited = 0

        while queue:
            # Collect all zero-in-degree nodes as one concurrent group
            group = list(queue)
            queue.clear()
            groups.append(group)
            visited += len(group)

            next_zero: List[str] = []
            for tid in group:
                for dep in dependents[tid]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_zero.append(dep)
            queue.extend(next_zero)

        if visited != len(self._tasks):
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Execution plan (public)
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute in parallel.

        Returns:
            List of task-ID groups in topological order.

        Raises:
            CircularDependencyError: If a cycle exists.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task) -> None:
        """Execute a single task with exponential-backoff retry logic.

        Args:
            task: The task to execute.
        """
        assert self._semaphore is not None

        m = self.metrics.tasks.setdefault(task.id, TaskMetrics(task_id=task.id))

        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            m.start_time = time.monotonic()

            for callback in self._on_task_start:
                callback(task)

            last_exc: Optional[Exception] = None

            while True:
                try:
                    if task._fn is None:
                        raise RuntimeError(f"Task '{task.id}' has no callable (_fn is None).")
                    task.result = await task._fn()
                    task.status = TaskStatus.COMPLETED
                    m.end_time = time.monotonic()
                    m.retry_count = task.retry_count

                    for callback in self._on_task_complete:
                        callback(task)
                    return

                except Exception as exc:
                    last_exc = exc
                    task.retry_count += 1

                    if task.retry_count > task.max_retries:
                        break

                    # Exponential backoff: 2^(retry-1) seconds, capped at 32 s
                    backoff = min(2 ** (task.retry_count - 1), 32)
                    await asyncio.sleep(backoff)

            # All retries exhausted
            task.status = TaskStatus.FAILED
            m.end_time = time.monotonic()
            m.retry_count = task.retry_count

            for callback in self._on_task_fail:
                callback(task, last_exc)  # type: ignore[arg-type]

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            :class:`SchedulerMetrics` populated with timing data.

        Raises:
            CircularDependencyError: If a cycle is detected.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        groups = self._topological_sort()
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

        self.metrics = SchedulerMetrics(total_start=time.monotonic())

        for group in groups:
            # Only schedule tasks whose dependencies all completed successfully
            runnable = [
                tid for tid in group
                if all(
                    self._tasks[dep].status == TaskStatus.COMPLETED
                    for dep in self._tasks[tid].dependencies
                )
            ]
            if runnable:
                await asyncio.gather(*(self._run_task(self._tasks[tid]) for tid in runnable))

        self.metrics.total_end = time.monotonic()
        return self.metrics
