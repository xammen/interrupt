"""
Async task scheduler with priority queue, dependency resolution,
concurrent execution, retry logic, and observer-pattern event emission.
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
    """Possible states of a scheduled task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when the dependency graph contains a cycle."""


@dataclass
class Task:
    """Represents a unit of work to be scheduled and executed."""

    id: str
    name: str
    priority: int  # 1 (lowest) – 10 (highest)
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


@dataclass
class TaskMetrics:
    """Execution metrics collected for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    retry_count: int = 0
    status: Optional[TaskStatus] = None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for an entire scheduler run."""

    total_time: Optional[float] = None
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_retries(self) -> int:
        return sum(m.retry_count for m in self.task_metrics.values())


# Type alias for the async callables the scheduler will invoke.
TaskCallable = Callable[[Task], Coroutine[Any, Any, Any]]

# Type alias for event listener callbacks.
EventListener = Callable[..., Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control,
    retry logic with exponential backoff, and an observer-pattern event system.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks that may run in parallel.  Defaults to 4.
    backoff_base:
        Base delay in seconds for exponential-backoff retries.  Defaults to 1.0.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        backoff_base: float = 1.0,
    ) -> None:
        self._tasks: dict[str, Task] = {}
        self._callables: dict[str, TaskCallable] = {}
        self._max_concurrency = max_concurrency
        self._backoff_base = backoff_base
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self._metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(self, task: Task, func: TaskCallable) -> None:
        """Register a task and its associated async callable.

        Parameters
        ----------
        task:
            The :class:`Task` instance to schedule.
        func:
            An async callable ``func(task) -> result`` that performs the work.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._callables[task.id] = func

    def remove_task(self, task_id: str) -> None:
        """Remove a task by its ID (only if it is still PENDING)."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task id '{task_id}'")
        if task.status != TaskStatus.PENDING:
            raise RuntimeError(
                f"Cannot remove task '{task_id}' with status {task.status.value}"
            )
        del self._tasks[task_id]
        self._callables.pop(task_id, None)

    # ------------------------------------------------------------------
    # Observer / event system
    # ------------------------------------------------------------------

    def on(self, event: str, listener: EventListener) -> None:
        """Subscribe *listener* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.
        """
        self._listeners[event].append(listener)

    def off(self, event: str, listener: EventListener) -> None:
        """Unsubscribe *listener* from *event*."""
        try:
            self._listeners[event].remove(listener)
        except ValueError:
            pass

    async def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event, calling every registered listener."""
        for listener in self._listeners.get(event, []):
            ret = listener(*args, **kwargs)
            if asyncio.iscoroutine(ret):
                await ret

    # ------------------------------------------------------------------
    # Dependency resolution (topological sort)
    # ------------------------------------------------------------------

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build an adjacency list and in-degree map from current tasks.

        Returns
        -------
        adjacency:
            ``{task_id: [dependent_task_ids …]}``
        in_degree:
            ``{task_id: number_of_unresolved_dependencies}``
        """
        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    raise KeyError(
                        f"Task '{tid}' depends on unknown task '{dep}'"
                    )
                adjacency[dep].append(tid)
                in_degree[tid] += 1

        return adjacency, in_degree

    def _topological_sort(self) -> list[list[str]]:
        """Kahn's algorithm producing *groups* of independent tasks.

        Each group contains tasks whose dependencies have all been satisfied
        by earlier groups.  Tasks within the same group can run concurrently.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        adjacency, in_degree = self._build_graph()
        queue: list[str] = [tid for tid, deg in in_degree.items() if deg == 0]
        # Sort within the initial frontier by descending priority.
        queue.sort(key=lambda t: self._tasks[t].priority, reverse=True)

        groups: list[list[str]] = []
        visited = 0

        while queue:
            # Everything currently in *queue* has in-degree 0 → independent.
            groups.append(list(queue))
            next_queue: list[str] = []
            for tid in queue:
                visited += 1
                for neighbor in adjacency[tid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            # Sort next frontier by priority (highest first).
            next_queue.sort(key=lambda t: self._tasks[t].priority, reverse=True)
            queue = next_queue

        if visited != len(self._tasks):
            # Find the nodes involved in the cycle for a helpful message.
            remaining = {tid for tid, deg in in_degree.items() if deg > 0}
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {remaining}"
            )

        return groups

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can be executed concurrently.
        Groups are ordered so that all dependencies of any task in group *n*
        have been placed in groups *< n*.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + exponential backoff."""
        func = self._callables[task.id]
        metric = TaskMetrics(task_id=task.id)
        self._metrics.task_metrics[task.id] = metric

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)
                metric.start_time = time.monotonic()

                try:
                    result = await func(task)
                    metric.end_time = time.monotonic()
                    metric.duration = metric.end_time - metric.start_time
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metric.status = TaskStatus.COMPLETED
                    metric.retry_count = task.retry_count
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    metric.end_time = time.monotonic()
                    task.retry_count += 1
                    metric.retry_count = task.retry_count

                    if task.retry_count >= task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.result = exc
                        metric.duration = metric.end_time - (metric.start_time or metric.end_time)
                        metric.status = TaskStatus.FAILED
                        await self._emit("on_task_fail", task, exc)
                        return

                    # Exponential backoff before next attempt.
                    delay = self._backoff_base * (2 ** (task.retry_count - 1))
                    task.status = TaskStatus.PENDING
                    await asyncio.sleep(delay)

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns
        -------
        SchedulerMetrics
            Collected timing and retry metrics for the run.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        self._metrics = SchedulerMetrics()
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()

        for group in groups:
            # Check that all dependencies for this group have completed.
            # If any dependency failed, mark dependent tasks as failed too.
            runnable: list[str] = []
            for tid in group:
                task = self._tasks[tid]
                failed_deps = [
                    d
                    for d in task.dependencies
                    if self._tasks[d].status == TaskStatus.FAILED
                ]
                if failed_deps:
                    task.status = TaskStatus.FAILED
                    task.result = RuntimeError(
                        f"Skipped because dependencies failed: {failed_deps}"
                    )
                    metric = TaskMetrics(
                        task_id=tid,
                        status=TaskStatus.FAILED,
                    )
                    self._metrics.task_metrics[tid] = metric
                    await self._emit("on_task_fail", task, task.result)
                else:
                    runnable.append(tid)

            # Run all independent tasks in this group concurrently.
            await asyncio.gather(
                *(self._run_task(self._tasks[tid], semaphore) for tid in runnable)
            )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Look up a task by its ID."""
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"Unknown task id '{task_id}'") from None

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of all registered tasks."""
        return dict(self._tasks)

    @property
    def metrics(self) -> SchedulerMetrics:
        """Metrics from the most recent :meth:`run` invocation."""
        return self._metrics
