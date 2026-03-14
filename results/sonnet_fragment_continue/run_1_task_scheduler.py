"""
task_scheduler.py
=================
Async task scheduler with priority queuing, dependency resolution,
concurrent execution, exponential backoff retries, and observer events.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


class DuplicateTaskError(Exception):
    """Raised when a task with the same ID is added more than once."""


class MissingDependencyError(Exception):
    """Raised when a task references a dependency that has not been registered."""


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
        retry_count: Number of retry attempts consumed so far.
        max_retries: Maximum number of retry attempts before marking as FAILED.
        created_at: UTC timestamp when the task was created.
        result: Value returned by the task coroutine on success.
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
    """Per-task timing and retry data collected during execution."""

    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds taken by the task (None if not finished)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for an entire scheduler run."""

    total_start: Optional[float] = None
    total_end: Optional[float] = None
    tasks: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_elapsed(self) -> Optional[float]:
        if self.total_start is not None and self.total_end is not None:
            return self.total_end - self.total_start
        return None


# ---------------------------------------------------------------------------
# Observer / event system
# ---------------------------------------------------------------------------

EventHandler = Callable[["Task"], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async observer that dispatches named events to registered handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        """Register *handler* to be called whenever *event* is emitted."""
        self._handlers[event].append(handler)

    async def emit(self, event: str, task: "Task") -> None:
        """Call all handlers registered for *event*, passing *task*."""
        for handler in self._handlers[event]:
            await handler(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Priority-aware async task scheduler with dependency resolution.

    Features
    --------
    * Topological sort for dependency ordering.
    * Circular dependency detection (raises :class:`CircularDependencyError`).
    * Concurrent execution within configurable ``max_concurrent`` limit.
    * Exponential backoff retry logic for failed tasks.
    * Observer pattern via :class:`EventBus` (``on_task_start``,
      ``on_task_complete``, ``on_task_fail``).
    * :meth:`get_execution_plan` returns ordered execution groups.
    * :attr:`metrics` exposes per-task and aggregate timing data.

    Parameters
    ----------
    max_concurrent:
        Maximum number of tasks that may run simultaneously.
        Defaults to ``5``.
    base_backoff:
        Base delay in seconds for exponential backoff.
        Retry *n* waits ``base_backoff * 2 ** (n - 1)`` seconds.
        Defaults to ``1.0``.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        base_backoff: float = 1.0,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self.max_concurrent = max_concurrent
        self.base_backoff = base_backoff
        self.events = EventBus()
        self.metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        coroutine_fn: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated async callable.

        Parameters
        ----------
        task:
            The :class:`Task` to schedule.
        coroutine_fn:
            An async callable (coroutine function) that performs the work.
            It receives no arguments; bind any needed context before
            passing it in.

        Raises
        ------
        DuplicateTaskError
            If a task with ``task.id`` has already been registered.
        """
        if task.id in self._tasks:
            raise DuplicateTaskError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_fn

    # ------------------------------------------------------------------
    # Dependency validation & topological sort
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency references a registered task.

        Raises
        ------
        MissingDependencyError
            If any task lists a dependency ID that has not been registered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise MissingDependencyError(
                        f"Task '{task.id}' depends on '{dep_id}', "
                        "which has not been registered."
                    )

    def _topological_sort(self) -> List[str]:
        """Return task IDs in a valid execution order (Kahn's algorithm).

        Within each topological layer tasks are sorted by descending priority
        so that higher-priority tasks are scheduled first.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.

        Returns
        -------
        List[str]
            Ordered list of task IDs.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Seed queue with tasks that have no dependencies, highest priority first.
        ready: deque[str] = deque(
            sorted(
                (tid for tid, deg in in_degree.items() if deg == 0),
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
        )

        order: List[str] = []
        while ready:
            tid = ready.popleft()
            order.append(tid)
            # Release dependents, inserting in priority order.
            newly_ready = []
            for dependent_id in dependents[tid]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    newly_ready.append(dependent_id)
            newly_ready.sort(key=lambda t: self._tasks[t].priority, reverse=True)
            ready.extend(newly_ready)

        if len(order) != len(self._tasks):
            cycle_ids = set(self._tasks) - set(order)
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_ids}"
            )

        return order

    # ------------------------------------------------------------------
    # Execution plan (public introspection helper)
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the execution plan as a list of parallel groups.

        Each inner list contains task IDs that can run concurrently
        (all their dependencies belong to earlier groups).  Tasks within
        a group are sorted by descending priority.

        Raises
        ------
        MissingDependencyError
            Forwarded from :meth:`_validate_dependencies`.
        CircularDependencyError
            Forwarded from :meth:`_topological_sort`.

        Returns
        -------
        List[List[str]]
            Ordered groups of task IDs.
        """
        self._validate_dependencies()

        # Compute the "level" (earliest group) for each task.
        level: Dict[str, int] = {}
        for tid in self._topological_sort():
            task = self._tasks[tid]
            if task.dependencies:
                level[tid] = max(level[dep] for dep in task.dependencies) + 1
            else:
                level[tid] = 0

        if not level:
            return []
        max_level = max(level.values(), default=0)
        groups: List[List[str]] = [[] for _ in range(max_level + 1)]
        for tid, lvl in level.items():
            groups[lvl].append(tid)

        # Sort each group by descending priority.
        for group in groups:
            group.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)

        return groups

    # ------------------------------------------------------------------
    # Core execution engine
    # ------------------------------------------------------------------

    async def _run_task_with_retry(self, task: Task) -> None:
        """Execute *task* with exponential backoff retries.

        On success the task's ``result`` is stored and its status set to
        ``COMPLETED``.  After exhausting ``max_retries`` the status is set
        to ``FAILED``.

        Events emitted
        --------------
        ``on_task_start``
            Fired once before the first attempt.
        ``on_task_complete``
            Fired on successful completion.
        ``on_task_fail``
            Fired after every failed attempt (including the final one).
        """
        tm = self.metrics.tasks.setdefault(task.id, TaskMetrics(task_id=task.id))

        task.status = TaskStatus.RUNNING
        tm.start_time = time.monotonic()
        await self.events.emit("on_task_start", task)

        coroutine_fn = self._coroutines[task.id]

        for attempt in range(task.max_retries + 1):
            try:
                task.result = await coroutine_fn()
                task.status = TaskStatus.COMPLETED
                tm.end_time = time.monotonic()
                tm.retry_count = task.retry_count
                await self.events.emit("on_task_complete", task)
                return
            except Exception:
                task.retry_count += 1
                tm.retry_count = task.retry_count
                await self.events.emit("on_task_fail", task)

                if attempt < task.max_retries:
                    backoff = self.base_backoff * (2 ** attempt)
                    await asyncio.sleep(backoff)

        # All attempts exhausted.
        task.status = TaskStatus.FAILED
        tm.end_time = time.monotonic()

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed in topological order.  Within each dependency
        layer, up to ``max_concurrent`` tasks run simultaneously.

        Returns
        -------
        SchedulerMetrics
            Timing and retry data for the completed run.

        Raises
        ------
        MissingDependencyError
            If any dependency ID is not registered.
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        order = self._topological_sort()

        self.metrics = SchedulerMetrics()
        self.metrics.total_start = time.monotonic()

        # Reset all task states for a fresh run.
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.result = None

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed: Set[str] = set()

        async def run_with_semaphore(task: Task) -> None:
            async with semaphore:
                await self._run_task_with_retry(task)
            completed.add(task.id)

        # Process tasks level by level so dependencies are always satisfied.
        groups = self.get_execution_plan()
        for group in groups:
            aws = [run_with_semaphore(self._tasks[tid]) for tid in group]
            await asyncio.gather(*aws)

        self.metrics.total_end = time.monotonic()
        return self.metrics

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[Task]:
        """Return the :class:`Task` with *task_id*, or ``None``."""
        return self._tasks.get(task_id)

    def pending_tasks(self) -> List[Task]:
        """Return all tasks currently in ``PENDING`` status."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def failed_tasks(self) -> List[Task]:
        """Return all tasks currently in ``FAILED`` status."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]

    def completed_tasks(self) -> List[Task]:
        """Return all tasks currently in ``COMPLETED`` status."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]
