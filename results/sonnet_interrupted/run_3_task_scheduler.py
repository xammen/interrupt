"""
task_scheduler.py

Async task scheduler with priority queues, dependency resolution,
concurrent execution, retry logic, and observer pattern event emission.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected among tasks."""


class TaskNotFoundError(KeyError):
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a single unit of work managed by the scheduler.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Scheduling priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of retries attempted so far.
        max_retries: Maximum number of retry attempts allowed.
        created_at: Timestamp of task creation.
        result: Return value of the task callable, populated on completion.
    """

    id: str
    name: str
    priority: int  # 1–10
    dependencies: List[str] = field(default_factory=list)
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
    """Execution metrics for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds from start to finish, or None if not complete."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a full scheduler run."""

    total_start: float = field(default_factory=time.monotonic)
    total_end: Optional[float] = None
    tasks: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrent execution.

    Features:
    - Priority queue: higher-priority tasks run first within each group.
    - Topological sort for dependency ordering; raises CircularDependencyError
      on cycles.
    - Configurable concurrency limit (semaphore-based).
    - Exponential backoff retry logic for failed tasks.
    - Observer pattern: register callbacks for task_start, task_complete,
      and task_fail events.

    Usage::

        scheduler = TaskScheduler(concurrency_limit=4)
        scheduler.add_task(task, coro_fn)
        results = await scheduler.run()
    """

    def __init__(self, concurrency_limit: int = 4) -> None:
        """
        Args:
            concurrency_limit: Maximum number of tasks that may run in parallel.
        """
        if concurrency_limit < 1:
            raise ValueError("concurrency_limit must be >= 1")

        self._concurrency_limit = concurrency_limit
        self._tasks: Dict[str, Task] = {}
        self._callables: Dict[str, Callable[..., Any]] = {}

        # Observer callbacks keyed by event name
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)

        # Populated during run()
        self._metrics: Optional[SchedulerMetrics] = None

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coro_fn: Callable[..., Any]) -> None:
        """Register a task and the coroutine function that implements it.

        Args:
            task: Task metadata object.
            coro_fn: An async callable (coroutine function) with no required
                     arguments.  It will be called as ``await coro_fn()``.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._callables[task.id] = coro_fn

    def on_task_start(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task begins execution."""
        self._listeners["task_start"].append(callback)

    def on_task_complete(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task finishes successfully."""
        self._listeners["task_complete"].append(callback)

    def on_task_fail(self, callback: Callable[[Task, Exception], Any]) -> None:
        """Register a callback invoked when a task fails all retries."""
        self._listeners["task_fail"].append(callback)

    # ------------------------------------------------------------------
    # Dependency / topology helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all dependency IDs reference registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Compute execution groups via Kahn's algorithm (BFS topological sort).

        Returns:
            A list of groups; each group is a list of task IDs that can run
            concurrently (all their dependencies are satisfied by earlier groups).
            Within each group tasks are ordered by descending priority.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._validate_dependencies()

        # in-degree count and reverse adjacency (who depends on me)
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            in_degree[task.id] += len(task.dependencies)
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)

        # Start with tasks that have no dependencies
        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: List[List[str]] = []
        visited = 0

        while queue:
            # Collect the current frontier as one execution group
            current_group = list(queue)
            queue.clear()

            # Sort group by descending priority so higher-priority tasks run first
            current_group.sort(
                key=lambda tid: self._tasks[tid].priority, reverse=True
            )
            groups.append(current_group)
            visited += len(current_group)

            next_wave: Set[str] = set()
            for tid in current_group:
                for dep in dependents[tid]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_wave.add(dep)
            queue.extend(next_wave)

        if visited != len(self._tasks):
            # Some nodes were never reached → cycle exists
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute in parallel; outer
        list order reflects the required execution sequence.

        Returns:
            List of lists of task IDs.

        Raises:
            CircularDependencyError: If a dependency cycle is detected.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def _emit(self, event: str, *args: Any) -> None:
        for cb in self._listeners[event]:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Task execution with retry / backoff
    # ------------------------------------------------------------------

    async def _execute_task(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
        metrics: SchedulerMetrics,
    ) -> None:
        """Execute a single task under the semaphore, with retry + backoff.

        Updates task.status, task.result, task.retry_count in place and
        records timing in *metrics*.
        """
        task_metrics = metrics.tasks.setdefault(task.id, TaskMetrics(task_id=task.id))
        coro_fn = self._callables[task.id]
        base_delay = 1.0  # seconds for exponential backoff

        async with semaphore:
            task.status = TaskStatus.RUNNING
            task_metrics.start_time = time.monotonic()
            await self._emit("task_start", task)

            last_exc: Optional[Exception] = None

            for attempt in range(task.max_retries + 1):
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    task.retry_count += 1
                    task_metrics.retry_count = task.retry_count

                try:
                    result = await coro_fn()
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task_metrics.end_time = time.monotonic()
                    await self._emit("task_complete", task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc

            # All attempts exhausted
            task.status = TaskStatus.FAILED
            task_metrics.end_time = time.monotonic()
            await self._emit("task_fail", task, last_exc)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A mapping of task ID → task result (None for failed tasks).

        Raises:
            CircularDependencyError: If a dependency cycle is detected.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        execution_groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self._concurrency_limit)
        self._metrics = SchedulerMetrics()

        for group in execution_groups:
            # Run all tasks in this group concurrently
            await asyncio.gather(
                *(
                    self._execute_task(self._tasks[tid], semaphore, self._metrics)
                    for tid in group
                )
            )

        self._metrics.total_end = time.monotonic()
        return {tid: task.result for tid, task in self._tasks.items()}

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Optional[SchedulerMetrics]:
        """Execution metrics from the most recent :meth:`run` call, or None."""
        return self._metrics
