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
# Enums & Exceptions
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable task name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of times the task has been retried.
        max_retries: Maximum retry attempts before marking the task as FAILED.
        created_at: Timestamp when the task was created.
        result: Value returned by the task coroutine on success.
    """

    id: str
    name: str
    priority: int  # 1–10
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
    """Execution metrics collected for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for a scheduler run."""

    total_start: Optional[float] = None
    total_end: Optional[float] = None
    per_task: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_time(self) -> Optional[float]:
        if self.total_start is not None and self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Observer / event callbacks type aliases
# ---------------------------------------------------------------------------

TaskCallback = Callable[[Task], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with priority queue, dependency resolution,
    concurrent execution limits, exponential-backoff retry, and event hooks.

    Args:
        concurrency_limit: Maximum number of tasks that may run simultaneously.
        base_retry_delay: Base delay in seconds for exponential-backoff retries.

    Example::

        async def handler(task: Task) -> Any:
            return await some_work(task)

        scheduler = TaskScheduler(concurrency_limit=4)
        scheduler.register(task, handler)
        await scheduler.run()
    """

    def __init__(
        self,
        concurrency_limit: int = 4,
        base_retry_delay: float = 1.0,
    ) -> None:
        self._concurrency_limit = concurrency_limit
        self._base_retry_delay = base_retry_delay

        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, TaskCallback] = {}

        # Observer callbacks
        self._on_task_start: List[TaskCallback] = []
        self._on_task_complete: List[TaskCallback] = []
        self._on_task_fail: List[TaskCallback] = []

        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, task: Task, handler: TaskCallback) -> None:
        """Register a task and its async handler coroutine.

        Args:
            task: The :class:`Task` instance to schedule.
            handler: An async callable ``async def handler(task) -> Any``
                     that performs the actual work.
        """
        self._tasks[task.id] = task
        self._handlers[task.id] = handler
        self.metrics.per_task[task.id] = TaskMetrics(task_id=task.id)

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------

    def on_task_start(self, callback: TaskCallback) -> None:
        """Register a callback invoked just before a task begins execution."""
        self._on_task_start.append(callback)

    def on_task_complete(self, callback: TaskCallback) -> None:
        """Register a callback invoked when a task completes successfully."""
        self._on_task_complete.append(callback)

    def on_task_fail(self, callback: TaskCallback) -> None:
        """Register a callback invoked when a task exhausts all retries and fails."""
        self._on_task_fail.append(callback)

    async def _emit(self, callbacks: List[TaskCallback], task: Task) -> None:
        for cb in callbacks:
            await cb(task)

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all declared dependency IDs refer to registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return execution groups ordered by dependency satisfaction.

        Each group contains task IDs that can run concurrently (all their
        dependencies are satisfied by previous groups).  Within a group,
        tasks are ordered from highest to lowest priority.

        Raises:
            CircularDependencyError: If a cycle exists in the dependency graph.

        Returns:
            List of groups, where each group is a list of task IDs.
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

        while queue:
            # Collect all zero-in-degree nodes as one concurrent group,
            # sorted by descending priority (higher priority first).
            current_group = sorted(
                list(queue),
                key=lambda tid: -self._tasks[tid].priority,
            )
            queue.clear()
            groups.append(current_group)

            next_wave: List[str] = []
            for tid in current_group:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_wave.append(dependent)
            queue.extend(next_wave)

        if sum(len(g) for g in groups) != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks: "
                + str([tid for tid, deg in in_degree.items() if deg > 0])
            )

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        Groups are ordered so that all dependencies of a group are satisfied
        by the groups that precede it.

        Raises:
            CircularDependencyError: If a cycle is detected.

        Returns:
            Ordered list of concurrent execution groups (lists of task IDs).
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task_with_retry(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task under the semaphore, retrying on failure.

        Uses exponential backoff: ``delay = base_retry_delay * 2 ** attempt``.

        Args:
            task: The task to execute.
            semaphore: Concurrency-limiting semaphore.
        """
        handler = self._handlers[task.id]
        m = self.metrics.per_task[task.id]

        async with semaphore:
            task.status = TaskStatus.RUNNING
            m.start_time = time.monotonic()
            await self._emit(self._on_task_start, task)

            attempt = 0
            while True:
                try:
                    task.result = await handler(task)
                    task.status = TaskStatus.COMPLETED
                    m.end_time = time.monotonic()
                    m.retry_count = task.retry_count
                    await self._emit(self._on_task_complete, task)
                    return
                except Exception:
                    attempt += 1
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        m.end_time = time.monotonic()
                        m.retry_count = task.retry_count
                        await self._emit(self._on_task_fail, task)
                        return
                    delay = self._base_retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

    async def run(self) -> None:
        """Execute all registered tasks respecting dependencies and concurrency limits.

        Tasks are executed group by group (topological order). Within each
        group, up to ``concurrency_limit`` tasks run simultaneously.

        Raises:
            CircularDependencyError: If a cycle is detected before execution.
        """
        self._validate_dependencies()
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self._concurrency_limit)

        self.metrics.total_start = time.monotonic()

        for group in groups:
            # Filter to only PENDING tasks (previous failures don't block others)
            pending = [
                tid for tid in group
                if self._tasks[tid].status == TaskStatus.PENDING
            ]
            if pending:
                await asyncio.gather(
                    *(
                        self._run_task_with_retry(self._tasks[tid], semaphore)
                        for tid in pending
                    )
                )

        self.metrics.total_end = time.monotonic()
