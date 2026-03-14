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
    # Internal helpers — event emission
    # ------------------------------------------------------------------

    def _emit_start(self, task: Task) -> None:
        for cb in self._on_task_start:
            cb(task)

    def _emit_complete(self, task: Task) -> None:
        for cb in self._on_task_complete:
            cb(task)

    def _emit_fail(self, task: Task, exc: Exception) -> None:
        for cb in self._on_task_fail:
            cb(task, exc)

    # ------------------------------------------------------------------
    # Internal helpers — dependency resolution (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency references a registered task ID.

        Raises:
            TaskNotFoundError: If a dependency task ID is not registered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[str]:
        """Return task IDs in a valid execution order via Kahn's algorithm.

        Tasks with no unresolved dependencies are sorted by descending priority
        so higher-priority work is surfaced first within each wave.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.

        Returns:
            Ordered list of task IDs.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Priority queue: use a list sorted by descending priority.
        ready: List[str] = [
            tid for tid, deg in in_degree.items() if deg == 0
        ]
        ready.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)

        order: List[str] = []
        while ready:
            # Pop from the front (highest priority among currently ready tasks)
            current = ready.pop(0)
            order.append(current)
            for dependent_id in dependents[current]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    ready.append(dependent_id)
                    # Re-sort to maintain priority ordering among newly ready tasks
                    ready.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)

        if len(order) != len(self._tasks):
            raise CircularDependencyError(
                "Circular dependency detected among tasks: "
                + str(set(self._tasks) - set(order))
            )

        return order

    # ------------------------------------------------------------------
    # Internal helpers — single task execution with retry
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task) -> None:
        """Execute a single task, retrying with exponential back-off on failure.

        Acquires the concurrency semaphore before executing and releases it
        afterwards. Updates :attr:`Task.status`, :attr:`Task.result`, and
        :attr:`Task.retry_count` in place.

        Args:
            task: The task to execute.
        """
        assert self._semaphore is not None  # set in run()

        task_metrics = self.metrics.tasks.setdefault(
            task.id, TaskMetrics(task_id=task.id)
        )

        last_exc: Optional[Exception] = None

        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            task_metrics.start_time = time.monotonic()
            self._emit_start(task)

            while task.retry_count <= task.max_retries:
                try:
                    if task._fn is None:
                        raise RuntimeError(
                            f"Task '{task.id}' has no callable (_fn is None)."
                        )
                    task.result = await task._fn()
                    task.status = TaskStatus.COMPLETED
                    task_metrics.end_time = time.monotonic()
                    task_metrics.retry_count = task.retry_count
                    self._emit_complete(task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        backoff = 2 ** (task.retry_count - 1)  # 1s, 2s, 4s …
                        await asyncio.sleep(backoff)
                    else:
                        break

            # Exhausted all retries
            task.status = TaskStatus.FAILED
            task_metrics.end_time = time.monotonic()
            task_metrics.retry_count = task.retry_count
            assert last_exc is not None
            self._emit_fail(task, last_exc)

    # ------------------------------------------------------------------
    # Public API — execution
    # ------------------------------------------------------------------

    async def run(self) -> SchedulerMetrics:
        """Validate, sort, and execute all registered tasks.

        Tasks are executed respecting dependency order and the
        ``max_concurrency`` limit. Independent tasks that are ready at the
        same wave are launched concurrently.

        Returns:
            :class:`SchedulerMetrics` populated with per-task and aggregate
            timing information.

        Raises:
            TaskNotFoundError: If any dependency references an unknown task.
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        execution_order = self._topological_sort()

        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self.metrics = SchedulerMetrics()
        self.metrics.total_start = time.monotonic()

        # Track which tasks have finished so we can gate dependents.
        completed: Set[str] = set()

        # Build a mapping from task_id -> set of task_ids that depend on it,
        # and in_degree per task for wave-based scheduling.
        dependents: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].add(task.id)

        ready: asyncio.Queue[str] = asyncio.Queue()
        for tid in execution_order:
            if in_degree[tid] == 0:
                await ready.put(tid)

        running: Dict[str, asyncio.Task[None]] = {}
        done_event = asyncio.Event()

        async def worker(task_id: str) -> None:
            task = self._tasks[task_id]
            await self._run_task(task)
            completed.add(task_id)
            # Unlock any newly ready dependents
            for dep_tid in dependents[task_id]:
                in_degree[dep_tid] -= 1
                if in_degree[dep_tid] == 0:
                    await ready.put(dep_tid)
            running.pop(task_id, None)
            if not running and ready.empty():
                done_event.set()

        total = len(self._tasks)

        async def dispatch() -> None:
            while len(completed) + len(running) < total or not ready.empty():
                try:
                    task_id = ready.get_nowait()
                except asyncio.QueueEmpty:
                    if len(completed) + len(running) >= total:
                        break
                    await asyncio.sleep(0)
                    continue
                t = asyncio.create_task(worker(task_id))
                running[task_id] = t

            # Wait for all in-flight tasks
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)

        await dispatch()

        self.metrics.total_end = time.monotonic()
        return self.metrics

    # ------------------------------------------------------------------
    # Public API — introspection
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Retrieve a registered task by ID.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The :class:`Task` instance.

        Raises:
            TaskNotFoundError: If no task with that ID is registered.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(f"No task registered with id '{task_id}'.")

    def get_all_tasks(self) -> List[Task]:
        """Return all registered tasks in insertion order."""
        return list(self._tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Return all tasks whose current status matches *status*.

        Args:
            status: The :class:`TaskStatus` value to filter by.

        Returns:
            List of matching :class:`Task` instances.
        """
        return [t for t in self._tasks.values() if t.status == status]

    def reset(self) -> None:
        """Reset all task statuses to PENDING and clear metrics.

        Useful for re-running the same scheduler after a previous :meth:`run`.
        """
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.result = None
        self.metrics = SchedulerMetrics()
