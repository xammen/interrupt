"""
task_scheduler.py - Async Task Scheduler with dependency resolution, retry logic,
concurrency control, and observer pattern event emission.
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
    """Lifecycle states for a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    Represents a unit of work managed by the scheduler.

    Attributes:
        id:           Unique identifier for the task.
        name:         Human-readable name.
        priority:     Execution priority (1 = lowest, 10 = highest).
        dependencies: IDs of tasks that must complete before this one runs.
        status:       Current lifecycle status.
        retry_count:  Number of times this task has been retried so far.
        max_retries:  Maximum number of allowed retries before marking FAILED.
        created_at:   Timestamp when the task was created.
        result:       Value returned by the coroutine on success.
    """
    id: str
    name: str
    priority: int                           # 1-10
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Any] = None


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Per-task timing and retry statistics collected during execution."""
    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds from start to finish, or *None* if not complete."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for a complete scheduler run."""
    total_time: float = 0.0
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observer helpers
# ---------------------------------------------------------------------------

EventCallback = Callable[[Task], None]


class EventEmitter:
    """Simple synchronous event emitter used by the scheduler."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

    def on(self, event: str, callback: EventCallback) -> None:
        """Register *callback* to be invoked whenever *event* is emitted."""
        self._listeners[event].append(callback)

    def emit(self, event: str, task: Task) -> None:
        """Invoke all callbacks registered for *event*, passing *task*."""
        for cb in self._listeners[event]:
            cb(task)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """
    Async task scheduler with dependency resolution, concurrency control,
    exponential-backoff retry logic, and observer-pattern event emission.

    Usage::

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(task, coroutine_fn)
        await scheduler.run()

    Events emitted (subscribe via ``scheduler.on(event, callback)``):
        - ``on_task_start``    – fired just before a task's coroutine starts.
        - ``on_task_complete`` – fired after a task finishes successfully.
        - ``on_task_fail``     – fired after a task exhausts all retries.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        """
        Args:
            max_concurrency: Maximum number of tasks that may run in parallel.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self._max_concurrency = max_concurrency
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[..., Any]] = {}
        self._emitter = EventEmitter()
        self._metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> None:
        """Register an observer callback for a named event."""
        self._emitter.on(event, callback)

    def add_task(
        self,
        task: Task,
        coroutine_fn: Callable[..., Any],
    ) -> None:
        """
        Register a task and its associated coroutine function.

        Args:
            task:         The :class:`Task` descriptor.
            coroutine_fn: An *async* callable (no arguments) that performs the
                          work.  It should return a value that will be stored in
                          ``task.result`` on success.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_fn

    def get_execution_plan(self) -> List[List[str]]:
        """
        Compute an ordered list of execution *groups*.

        Tasks within the same group have no inter-dependencies and can run
        concurrently.  Groups are ordered so that all dependencies of group *N*
        appear in groups 0 … N-1.

        Returns:
            A list of groups, where each group is a list of task IDs sorted by
            descending priority (highest priority first).

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            TaskNotFoundError:       If a task references an unknown dependency.
        """
        self._validate_dependencies()
        return self._topological_groups()

    @property
    def metrics(self) -> SchedulerMetrics:
        """Execution metrics accumulated during the last :meth:`run` call."""
        return self._metrics

    async def run(self) -> Dict[str, Any]:
        """
        Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A mapping of ``task_id -> task.result`` for every completed task.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            TaskNotFoundError:       If a task references an unknown dependency.
        """
        self._metrics = SchedulerMetrics()
        plan = self.get_execution_plan()
        scheduler_start = time.monotonic()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in plan:
            # Sort within the group by priority descending so the semaphore
            # favours high-priority tasks when the limit is in play.
            sorted_group = sorted(
                group, key=lambda tid: self._tasks[tid].priority, reverse=True
            )
            await asyncio.gather(
                *(self._execute_task(tid, semaphore) for tid in sorted_group)
            )

        self._metrics.total_time = time.monotonic() - scheduler_start
        return {tid: t.result for tid, t in self._tasks.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency references a known task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_groups(self) -> List[List[str]]:
        """
        Kahn's algorithm – returns execution groups (level-order BFS).

        Raises:
            CircularDependencyError: When a cycle is detected.
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
            # All nodes currently at in-degree 0 form one parallel group.
            current_group = list(queue)
            queue.clear()
            groups.append(current_group)
            visited += len(current_group)

            for tid in current_group:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if visited != len(self._tasks):
            cycle_nodes = [tid for tid, d in in_degree.items() if d > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    async def _execute_task(
        self, task_id: str, semaphore: asyncio.Semaphore
    ) -> None:
        """
        Run a single task under the semaphore, retrying with exponential backoff.

        Back-off delay formula: ``2 ** attempt`` seconds (capped at 32 s).
        """
        task = self._tasks[task_id]
        metrics = TaskMetrics(task_id=task_id)
        self._metrics.task_metrics[task_id] = metrics

        async with semaphore:
            metrics.start_time = time.monotonic()
            task.status = TaskStatus.RUNNING
            self._emitter.emit("on_task_start", task)

            last_exception: Optional[Exception] = None

            for attempt in range(task.max_retries + 1):
                try:
                    coro_fn = self._coroutines[task_id]
                    task.result = await coro_fn()
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    self._emitter.emit("on_task_complete", task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc
                    if attempt < task.max_retries:
                        task.retry_count += 1
                        delay = min(2 ** attempt, 32)
                        await asyncio.sleep(delay)
                    # else: fall through to FAILED

            task.status = TaskStatus.FAILED
            metrics.end_time = time.monotonic()
            metrics.retry_count = task.retry_count
            self._emitter.emit("on_task_fail", task)
            # Re-raise so the caller can inspect if needed; gather() will
            # surface it but we intentionally continue the rest of the plan.
            raise RuntimeError(
                f"Task '{task_id}' failed after {task.max_retries} retries."
            ) from last_exception
