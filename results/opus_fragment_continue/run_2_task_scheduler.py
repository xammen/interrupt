"""
Async task scheduler with dependency resolution, retry logic, and concurrent execution.

Provides a TaskScheduler that manages tasks with priorities, resolves dependency
graphs via topological sort, detects circular dependencies, runs independent tasks
concurrently (with a configurable concurrency limit), and implements exponential
backoff retry logic for failed tasks. Events are emitted via an observer pattern.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected among tasks."""

    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    """Possible states of a scheduled task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a unit of work to be scheduled.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: How many times this task has been retried so far.
        max_retries: Maximum number of retry attempts before marking as FAILED.
        created_at: Timestamp when the task was created.
        result: The return value of the task's coroutine, if completed.
    """

    id: str
    name: str
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"Priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------


@dataclass
class TaskMetrics:
    """Execution metrics for a single task."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retries: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Wall-clock duration in seconds, or None if not yet finished."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for a scheduler run."""

    total_start: Optional[float] = None
    total_end: Optional[float] = None
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_time(self) -> Optional[float]:
        """Total wall-clock time for the entire run."""
        if self.total_start is not None and self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Event types used by the observer pattern
# ---------------------------------------------------------------------------

EventCallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Features:
        - Priority-aware scheduling (higher priority runs first within a group).
        - Topological-sort-based dependency resolution with cycle detection.
        - Configurable concurrency limit for parallel task execution.
        - Exponential-backoff retry logic for transient failures.
        - Observer-pattern event emission (on_task_start, on_task_complete, on_task_fail).
        - Execution metrics tracking.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
            Defaults to 4.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)
        self._metrics = SchedulerMetrics()

    # -- Task management ----------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The Task instance to schedule.
            coro_func: An async callable that performs the task's work.
                       It receives the Task as its sole argument.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro_func

    def get_task(self, task_id: str) -> Task:
        """Return the task with the given ID.

        Raises:
            TaskNotFoundError: If no task with that ID exists.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task with id '{task_id}'")

    @property
    def metrics(self) -> SchedulerMetrics:
        """Access the execution metrics collected during the last run."""
        return self._metrics

    # -- Observer pattern ---------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> None:
        """Subscribe *callback* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.
        """
        self._listeners[event].append(callback)

    async def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all callbacks registered for *event*."""
        for cb in self._listeners.get(event, []):
            result = cb(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result

    # -- Dependency resolution ----------------------------------------------

    def _validate_dependencies(self) -> None:
        """Check that every dependency reference points to a known task.

        Raises:
            TaskNotFoundError: If a task lists a dependency ID that hasn't
                been registered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_cycle(self) -> None:
        """Detect circular dependencies using iterative DFS with three-color marking.

        Raises:
            CircularDependencyError: If a cycle is found, includes the cycle path.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: Dict[str, Optional[str]] = {tid: None for tid in self._tasks}

        for start in self._tasks:
            if color[start] != WHITE:
                continue
            stack: List[tuple[str, int]] = [(start, 0)]
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx == 0:
                    color[node] = GRAY
                if idx < len(deps):
                    stack.append((node, idx + 1))
                    dep = deps[idx]
                    if color[dep] == GRAY:
                        # Reconstruct cycle
                        cycle = [dep, node]
                        cur = node
                        while cur != dep:
                            cur = parent[cur]  # type: ignore[assignment]
                            if cur is None:
                                break
                            cycle.append(cur)
                        cycle.reverse()
                        raise CircularDependencyError(cycle)
                    if color[dep] == WHITE:
                        parent[dep] = node
                        stack.append((dep, 0))
                else:
                    color[node] = BLACK

    def _topological_sort(self) -> List[List[str]]:
        """Return tasks grouped into layers by topological order.

        Each layer is a list of task IDs that can run concurrently (all their
        dependencies appear in earlier layers).  Within each layer tasks are
        sorted by descending priority so higher-priority tasks start first.

        Returns:
            A list of layers, where each layer is a list of task IDs.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task.
            CircularDependencyError: If a circular dependency exists.
        """
        self._validate_dependencies()
        self._detect_cycle()

        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Kahn's algorithm, collecting layers
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        layers: List[List[str]] = []

        while queue:
            # Sort current layer by priority descending
            queue.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            layers.append(list(queue))
            next_queue: List[str] = []
            for tid in queue:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            queue = next_queue

        return layers

    # -- Retry logic --------------------------------------------------------

    @staticmethod
    def _backoff_delay(attempt: int, base: float = 0.1, cap: float = 5.0) -> float:
        """Calculate exponential backoff delay with a cap.

        Args:
            attempt: The current retry attempt number (0-based).
            base: Base delay in seconds.
            cap: Maximum delay in seconds.

        Returns:
            The delay in seconds before the next retry.
        """
        delay = min(base * (2 ** attempt), cap)
        return delay

    # -- Single task execution ----------------------------------------------

    async def _run_task(self, task_id: str, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry logic, guarded by *semaphore*.

        On each attempt the coroutine registered for *task_id* is awaited.
        If it raises, exponential backoff is applied and the task is retried
        up to ``task.max_retries`` times.  Metrics and events are recorded.

        Args:
            task_id: ID of the task to execute.
            semaphore: Semaphore that enforces the concurrency limit.
        """
        task = self._tasks[task_id]
        coro_func = self._coroutines[task_id]
        metrics = TaskMetrics(task_id=task_id)
        self._metrics.task_metrics[task_id] = metrics

        async with semaphore:
            task.status = TaskStatus.RUNNING
            await self._emit("on_task_start", task)
            metrics.start_time = time.monotonic()

            while True:
                try:
                    task.result = await coro_func(task)
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    task.retry_count += 1
                    metrics.retries = task.retry_count
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        metrics.end_time = time.monotonic()
                        await self._emit("on_task_fail", task, exc)
                        return
                    delay = self._backoff_delay(task.retry_count - 1)
                    await asyncio.sleep(delay)

    # -- Main entry point ---------------------------------------------------

    async def run(self) -> Dict[str, Task]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A dict mapping task IDs to their (now-updated) Task objects.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task.
            CircularDependencyError: If a circular dependency exists.
        """
        if not self._tasks:
            return {}

        self._metrics = SchedulerMetrics()
        self._metrics.total_start = time.monotonic()

        layers = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for layer in layers:
            # Check that all dependencies completed successfully
            failed_deps: Dict[str, List[str]] = {}
            for tid in layer:
                task = self._tasks[tid]
                for dep_id in task.dependencies:
                    dep_task = self._tasks[dep_id]
                    if dep_task.status == TaskStatus.FAILED:
                        failed_deps.setdefault(tid, []).append(dep_id)

            runnable: List[str] = []
            for tid in layer:
                if tid in failed_deps:
                    task = self._tasks[tid]
                    task.status = TaskStatus.FAILED
                    exc = RuntimeError(
                        f"Dependency failed: {', '.join(failed_deps[tid])}"
                    )
                    await self._emit("on_task_fail", task, exc)
                else:
                    runnable.append(tid)

            # Run all runnable tasks in this layer concurrently
            if runnable:
                await asyncio.gather(
                    *(self._run_task(tid, semaphore) for tid in runnable)
                )

        self._metrics.total_end = time.monotonic()
        return dict(self._tasks)

    # -- Utility helpers ----------------------------------------------------

    def reset(self) -> None:
        """Reset all tasks to PENDING and clear metrics.

        Useful for re-running the same scheduler instance.
        """
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.result = None
        self._metrics = SchedulerMetrics()

    def remove_task(self, task_id: str) -> Task:
        """Remove and return a task by ID.

        Also removes any references to *task_id* from other tasks' dependency
        lists.

        Raises:
            TaskNotFoundError: If no task with that ID exists.
        """
        task = self.get_task(task_id)  # raises if missing
        del self._tasks[task_id]
        del self._coroutines[task_id]
        # Clean up dependency references
        for other in self._tasks.values():
            if task_id in other.dependencies:
                other.dependencies.remove(task_id)
        return task

    @property
    def task_count(self) -> int:
        """Return the number of registered tasks."""
        return len(self._tasks)

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Return all tasks with the given status."""
        return [t for t in self._tasks.values() if t.status == status]
