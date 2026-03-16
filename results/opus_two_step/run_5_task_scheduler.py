"""
Async task scheduler with dependency resolution, retry logic, and observer pattern.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class TaskStatus(Enum):
    """Status of a task in the scheduler."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""
    pass


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of times the task has been retried.
        max_retries: Maximum retry attempts before permanent failure.
        created_at: Timestamp when the task was created.
        result: The return value of the task after completion, or None.
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
            raise ValueError(f"Priority must be between 1 and 10, got {self.priority}")


@dataclass
class TaskMetrics:
    """Execution metrics for the scheduler.

    Attributes:
        total_time: Wall-clock time for the entire execution run.
        per_task_time: Mapping of task ID to its execution duration in seconds.
        retry_counts: Mapping of task ID to the number of retries performed.
    """
    total_time: float = 0.0
    per_task_time: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control,
    retry logic, and an observer-based event system.

    Args:
        max_concurrency: Maximum number of tasks to run in parallel.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: dict[str, Task] = {}
        self._task_functions: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._max_concurrency = max_concurrency
        self._metrics = TaskMetrics()

    # ------------------------------------------------------------------ #
    # Task registration
    # ------------------------------------------------------------------ #

    def add_task(
        self,
        task: Task,
        func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The Task instance to schedule.
            func: An async callable that performs the task's work.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._task_functions[task.id] = func

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the scheduler.

        Args:
            task_id: The ID of the task to remove.

        Raises:
            KeyError: If no task with the given ID exists.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        del self._tasks[task_id]
        del self._task_functions[task_id]

    # ------------------------------------------------------------------ #
    # Observer / event system
    # ------------------------------------------------------------------ #

    def on_task_start(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task begins execution.

        Args:
            callback: A callable that receives the Task instance.
        """
        self._listeners["on_task_start"].append(callback)

    def on_task_complete(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task completes successfully.

        Args:
            callback: A callable that receives the Task instance.
        """
        self._listeners["on_task_complete"].append(callback)

    def on_task_fail(self, callback: Callable[[Task, Exception], Any]) -> None:
        """Register a callback invoked when a task fails (after all retries).

        Args:
            callback: A callable that receives the Task instance and the exception.
        """
        self._listeners["on_task_fail"].append(callback)

    async def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners.

        Args:
            event: The event name.
            *args: Arguments forwarded to each listener.
        """
        for callback in self._listeners.get(event, []):
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------ #
    # Dependency resolution (topological sort)
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from current tasks.

        Returns:
            A tuple of (adjacency list, in-degree mapping).

        Raises:
            KeyError: If a dependency references an unknown task ID.
        """
        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise KeyError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )
                adjacency[dep_id].append(task.id)
                in_degree[task.id] += 1

        return adjacency, in_degree

    def _topological_sort(self) -> list[list[str]]:
        """Perform a topological sort grouped into concurrency layers.

        Each inner list contains task IDs that may be executed in parallel
        (they share the same topological depth).

        Returns:
            Ordered list of execution groups.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        adjacency, in_degree = self._build_graph()

        queue: deque[str] = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)

        groups: list[list[str]] = []
        visited_count = 0

        while queue:
            # All tasks currently in the queue form one execution group.
            # Sort within each group by descending priority for determinism.
            group = sorted(queue, key=lambda t: self._tasks[t].priority, reverse=True)
            groups.append(group)
            visited_count += len(group)

            next_queue: deque[str] = deque()
            for tid in group:
                for neighbor in adjacency[tid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        if visited_count != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks: unable to produce a "
                "complete topological ordering."
            )

        return groups

    # ------------------------------------------------------------------ #
    # Execution plan
    # ------------------------------------------------------------------ #

    def get_execution_plan(self) -> list[list[Task]]:
        """Return the ordered execution groups resolved from the dependency graph.

        Each inner list represents a group of tasks that can run concurrently.
        Groups are ordered so that all dependencies of every task in group *i*
        appear in groups *< i*.

        Returns:
            A list of task groups (lists of Task instances).

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            KeyError: If a dependency references an unknown task ID.
        """
        groups = self._topological_sort()
        return [[self._tasks[tid] for tid in group] for group in groups]

    # ------------------------------------------------------------------ #
    # Task execution with retry
    # ------------------------------------------------------------------ #

    async def _run_task(self, task: Task) -> None:
        """Execute a single task with exponential-backoff retry.

        Args:
            task: The task to execute.
        """
        func = self._task_functions[task.id]
        task.status = TaskStatus.RUNNING
        await self._emit("on_task_start", task)

        start = time.monotonic()
        last_exception: Optional[Exception] = None

        while task.retry_count <= task.max_retries:
            try:
                task.result = await func()
                task.status = TaskStatus.COMPLETED
                elapsed = time.monotonic() - start
                self._metrics.per_task_time[task.id] = elapsed
                self._metrics.retry_counts[task.id] = task.retry_count
                await self._emit("on_task_complete", task)
                return
            except Exception as exc:
                last_exception = exc
                if task.retry_count < task.max_retries:
                    backoff = 2 ** task.retry_count * 0.1  # 0.1s, 0.2s, 0.4s …
                    await asyncio.sleep(backoff)
                    task.retry_count += 1
                else:
                    break

        # All retries exhausted
        task.status = TaskStatus.FAILED
        elapsed = time.monotonic() - start
        self._metrics.per_task_time[task.id] = elapsed
        self._metrics.retry_counts[task.id] = task.retry_count
        assert last_exception is not None
        await self._emit("on_task_fail", task, last_exception)

    # ------------------------------------------------------------------ #
    # Main execution loop
    # ------------------------------------------------------------------ #

    async def run(self) -> TaskMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed in topological order; within each dependency layer
        they run concurrently up to ``max_concurrency`` parallel tasks.
        Higher-priority tasks within a group are started first.

        Returns:
            A :class:`TaskMetrics` instance summarising the run.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            KeyError: If a dependency references an unknown task ID.
        """
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _limited_run(task: Task) -> None:
            async with semaphore:
                await self._run_task(task)

        overall_start = time.monotonic()

        for group in groups:
            # Only run tasks that are still PENDING (skip already failed upstream deps).
            runnable = [
                self._tasks[tid]
                for tid in group
                if self._tasks[tid].status == TaskStatus.PENDING
                and all(
                    self._tasks[dep].status == TaskStatus.COMPLETED
                    for dep in self._tasks[tid].dependencies
                )
            ]

            # Mark tasks whose dependencies failed so they don't hang.
            for tid in group:
                task = self._tasks[tid]
                if task.status == TaskStatus.PENDING and task not in runnable:
                    task.status = TaskStatus.FAILED
                    self._metrics.per_task_time[task.id] = 0.0
                    self._metrics.retry_counts[task.id] = 0
                    await self._emit(
                        "on_task_fail",
                        task,
                        RuntimeError(
                            f"Skipped: one or more dependencies of '{task.id}' failed"
                        ),
                    )

            await asyncio.gather(*(_limited_run(t) for t in runnable))

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    @property
    def metrics(self) -> TaskMetrics:
        """Return the most recent execution metrics."""
        return self._metrics

    @property
    def tasks(self) -> dict[str, Task]:
        """Return the internal task registry (read-only view)."""
        return dict(self._tasks)
