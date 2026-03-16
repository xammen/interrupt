"""Async task scheduler with dependency resolution, retry logic, and observer pattern."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class TaskStatus(Enum):
    """Possible states of a task."""
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
        name: Human-readable task name.
        priority: Execution priority from 1 (lowest) to 10 (highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum retry attempts before permanent failure.
        created_at: Timestamp of task creation.
        result: The return value of the task's coroutine, if completed.
    """
    id: str
    name: str
    priority: int
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
    """Execution metrics for a single task.

    Attributes:
        task_id: The task this metric belongs to.
        start_time: When execution started (seconds since epoch).
        end_time: When execution finished (seconds since epoch).
        duration: Wall-clock duration in seconds.
        retries: Number of retries that occurred.
        success: Whether the task ultimately succeeded.
    """
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    retries: int = 0
    success: bool = False


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for an entire scheduler run.

    Attributes:
        total_time: Wall-clock time for the full execution.
        task_metrics: Per-task metrics keyed by task ID.
        total_retries: Sum of all retries across tasks.
    """
    total_time: float = 0.0
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)
    total_retries: int = 0


EventCallback = Callable[["Task"], Coroutine[Any, Any, None] | None]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control, and retry logic.

    The scheduler accepts tasks with explicit dependency declarations, performs
    topological sorting to determine execution order, and runs independent tasks
    concurrently up to a configurable limit. Failed tasks are retried with
    exponential backoff. Observers can subscribe to lifecycle events.

    Args:
        max_concurrency: Maximum number of tasks to run in parallel.
        base_backoff: Base delay in seconds for exponential backoff on retries.
    """

    def __init__(self, max_concurrency: int = 4, base_backoff: float = 1.0) -> None:
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency: int = max_concurrency
        self._base_backoff: float = base_backoff
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._metrics: SchedulerMetrics = SchedulerMetrics()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)

    # ── Task registration ───────────────────────────────────────────────

    def add_task(
        self,
        task: Task,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The task descriptor.
            coro_func: An async callable that performs the actual work.
                       It receives the :class:`Task` instance as its sole argument.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro_func

    def remove_task(self, task_id: str) -> None:
        """Remove a task that has not yet started.

        Args:
            task_id: ID of the task to remove.

        Raises:
            KeyError: If the task does not exist.
            RuntimeError: If the task is currently running.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        if task.status == TaskStatus.RUNNING:
            raise RuntimeError(f"Cannot remove running task '{task_id}'")
        del self._tasks[task_id]
        del self._coroutines[task_id]

    # ── Observer pattern ────────────────────────────────────────────────

    def on_task_start(self, callback: EventCallback) -> None:
        """Register a listener invoked when a task begins execution."""
        self._listeners["on_task_start"].append(callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Register a listener invoked when a task completes successfully."""
        self._listeners["on_task_complete"].append(callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Register a listener invoked when a task fails permanently."""
        self._listeners["on_task_fail"].append(callback)

    async def _emit(self, event: str, task: Task) -> None:
        """Dispatch an event to all registered listeners.

        Args:
            event: The event name (e.g. ``'on_task_start'``).
            task: The task associated with the event.
        """
        for callback in self._listeners.get(event, []):
            ret = callback(task)
            if asyncio.iscoroutine(ret):
                await ret

    # ── Dependency resolution ───────────────────────────────────────────

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from registered tasks.

        Returns:
            A tuple of (adjacency list, in-degree map).

        Raises:
            KeyError: If a dependency references a task that has not been added.
        """
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise KeyError(
                        f"Task '{tid}' depends on unknown task '{dep_id}'"
                    )
                adj[dep_id].append(tid)
                in_degree[tid] += 1

        return adj, in_degree

    def _topological_sort(self) -> list[list[str]]:
        """Perform a topological sort, grouping tasks into concurrent execution layers.

        Tasks in the same layer have no mutual dependencies and can run in
        parallel.  Within each layer tasks are sorted by descending priority.

        Returns:
            Ordered list of execution groups (each group is a list of task IDs).

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        adj, in_degree = self._build_graph()

        queue: deque[str] = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)

        layers: list[list[str]] = []
        visited_count = 0

        while queue:
            layer: list[str] = []
            for _ in range(len(queue)):
                tid = queue.popleft()
                layer.append(tid)
                visited_count += 1

            # Sort layer by descending priority
            layer.sort(key=lambda t: self._tasks[t].priority, reverse=True)
            layers.append(layer)

            next_queue: deque[str] = deque()
            for tid in layer:
                for neighbour in adj[tid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        next_queue.append(neighbour)
            queue = next_queue

        if visited_count != len(self._tasks):
            # Identify the tasks involved in the cycle for a helpful message
            cycle_tasks = [
                tid for tid, deg in in_degree.items() if deg > 0
            ]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_tasks}"
            )

        return layers

    # ── Execution plan ──────────────────────────────────────────────────

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that may execute concurrently.
        Groups are ordered so that all dependencies of a group are satisfied
        by earlier groups.

        Returns:
            List of execution groups.

        Raises:
            CircularDependencyError: If the graph contains a cycle.
            KeyError: If a dependency references an unknown task.
        """
        return self._topological_sort()

    # ── Task execution ──────────────────────────────────────────────────

    async def _run_task(self, task_id: str) -> None:
        """Execute a single task with retry and exponential backoff.

        Args:
            task_id: ID of the task to execute.
        """
        task = self._tasks[task_id]
        coro_func = self._coroutines[task_id]
        metrics = TaskMetrics(task_id=task_id)

        while True:
            async with self._semaphore:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)
                metrics.start_time = time.monotonic()

                try:
                    task.result = await coro_func(task)
                    metrics.end_time = time.monotonic()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.retries = task.retry_count
                    metrics.success = True
                    task.status = TaskStatus.COMPLETED
                    await self._emit("on_task_complete", task)
                    self._metrics.task_metrics[task_id] = metrics
                    return

                except Exception:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        metrics.end_time = time.monotonic()
                        metrics.duration = metrics.end_time - metrics.start_time
                        metrics.retries = task.retry_count - 1
                        metrics.success = False
                        task.status = TaskStatus.FAILED
                        await self._emit("on_task_fail", task)
                        self._metrics.task_metrics[task_id] = metrics
                        return

            # Exponential backoff (outside the semaphore so we don't hold it)
            backoff = self._base_backoff * (2 ** (task.retry_count - 1))
            await asyncio.sleep(backoff)

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Independent tasks within the same execution layer run concurrently,
        bounded by the scheduler's concurrency limit.  Layers execute
        sequentially so that all dependencies are satisfied.

        Returns:
            :class:`SchedulerMetrics` with timing and retry information.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            KeyError: If a dependency references an unknown task.
        """
        self._metrics = SchedulerMetrics()
        execution_plan = self._topological_sort()

        overall_start = time.monotonic()

        for layer in execution_plan:
            # Filter out tasks whose dependencies failed
            runnable: list[str] = []
            for tid in layer:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_ok:
                    runnable.append(tid)
                else:
                    task.status = TaskStatus.FAILED
                    task.result = None
                    metrics = TaskMetrics(
                        task_id=tid,
                        success=False,
                    )
                    self._metrics.task_metrics[tid] = metrics
                    await self._emit("on_task_fail", task)

            if runnable:
                await asyncio.gather(*(self._run_task(tid) for tid in runnable))

        overall_end = time.monotonic()
        self._metrics.total_time = overall_end - overall_start
        self._metrics.total_retries = sum(
            m.retries for m in self._metrics.task_metrics.values()
        )

        return self._metrics

    # ── Utility ─────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Task:
        """Retrieve a registered task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            The :class:`Task` instance.

        Raises:
            KeyError: If no task with that ID exists.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"Task '{task_id}' not found")

    @property
    def metrics(self) -> SchedulerMetrics:
        """Access the most recent execution metrics."""
        return self._metrics

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of all registered tasks."""
        return dict(self._tasks)
