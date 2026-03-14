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

    # -- Dependency resolution (topological sort) ---------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency reference points to a known task.

        Raises:
            TaskNotFoundError: If a dependency ID is not registered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_cycle(self) -> None:
        """Detect circular dependencies using iterative DFS with color marking.

        Raises:
            CircularDependencyError: If a cycle is found.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: Dict[str, Optional[str]] = {tid: None for tid in self._tasks}

        for start in self._tasks:
            if color[start] != WHITE:
                continue
            stack: List[tuple[str, int]] = [(start, 0)]
            color[start] = GRAY
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx < len(deps):
                    # Push the current node back with next index
                    stack.append((node, idx + 1))
                    dep = deps[idx]
                    if color[dep] == GRAY:
                        # Reconstruct cycle path
                        cycle = [dep, node]
                        for s_node, _ in reversed(stack):
                            if s_node == dep:
                                break
                            cycle.append(s_node)
                        cycle.append(dep)
                        cycle.reverse()
                        raise CircularDependencyError(cycle)
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        parent[dep] = node
                        stack.append((dep, 0))
                else:
                    color[node] = BLACK

    def _topological_groups(self) -> List[List[str]]:
        """Return task IDs grouped into topological layers (Kahn's algorithm).

        Tasks within the same layer have no mutual dependencies and may run
        concurrently.  Within each layer, tasks are sorted by descending
        priority so higher-priority tasks start first.

        Returns:
            A list of layers, each layer being a list of task IDs.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Seed with zero-in-degree tasks, sorted by priority (desc)
        queue = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda tid: self._tasks[tid].priority,
            reverse=True,
        )

        groups: List[List[str]] = []
        while queue:
            groups.append(list(queue))
            next_queue: List[str] = []
            for tid in queue:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            next_queue.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            queue = next_queue

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute in parallel.
        The outer list is ordered: group *i* must finish before group *i+1*
        starts.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task.
            CircularDependencyError: If a cycle exists in the dependency graph.

        Returns:
            Ordered list of parallel execution groups.
        """
        self._validate_dependencies()
        self._detect_cycle()
        return self._topological_groups()

    # -- Execution ----------------------------------------------------------

    async def _run_single_task(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task with retry / exponential-backoff logic.

        Args:
            task: The task to run.
            semaphore: Controls concurrency.
        """
        metrics = TaskMetrics(task_id=task.id)
        self._metrics.task_metrics[task.id] = metrics

        async with semaphore:
            metrics.start_time = time.monotonic()
            attempt = 0
            while True:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)
                try:
                    result = await self._coroutines[task.id](task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retries = task.retry_count
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    attempt += 1
                    task.retry_count = attempt
                    if attempt > task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.result = exc
                        metrics.end_time = time.monotonic()
                        metrics.retries = task.retry_count
                        await self._emit("on_task_fail", task, exc)
                        return
                    # Exponential backoff: 2^(attempt-1) * 0.1 seconds
                    backoff = (2 ** (attempt - 1)) * 0.1
                    await asyncio.sleep(backoff)

    async def run(self) -> Dict[str, Task]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A dict mapping task IDs to their (now-updated) Task objects.

        Raises:
            TaskNotFoundError: If a dependency references an unknown task.
            CircularDependencyError: If a cycle exists in the dependency graph.
        """
        self._validate_dependencies()
        self._detect_cycle()
        groups = self._topological_groups()

        semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics = SchedulerMetrics()
        self._metrics.total_start = time.monotonic()

        for group in groups:
            # Skip tasks already marked FAILED by cascade from a prior group
            runnable = [
                tid for tid in group
                if self._tasks[tid].status != TaskStatus.FAILED
            ]
            coros = [
                self._run_single_task(self._tasks[tid], semaphore) for tid in runnable
            ]
            await asyncio.gather(*coros)

            # If any dependency in this group failed, mark downstream as failed
            failed_ids: Set[str] = {
                tid
                for tid in group
                if self._tasks[tid].status == TaskStatus.FAILED
            }
            if failed_ids:
                self._cascade_failure(failed_ids)

        self._metrics.total_end = time.monotonic()
        return dict(self._tasks)

    def _cascade_failure(self, failed_ids: Set[str]) -> None:
        """Mark all tasks that transitively depend on *failed_ids* as FAILED."""
        dependents: Dict[str, List[str]] = defaultdict(list)
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)

        queue = list(failed_ids)
        visited: Set[str] = set(failed_ids)
        while queue:
            current = queue.pop(0)
            for dep_tid in dependents.get(current, []):
                if dep_tid not in visited:
                    self._tasks[dep_tid].status = TaskStatus.FAILED
                    self._tasks[dep_tid].result = RuntimeError(
                        f"Dependency '{current}' failed"
                    )
                    visited.add(dep_tid)
                    queue.append(dep_tid)
