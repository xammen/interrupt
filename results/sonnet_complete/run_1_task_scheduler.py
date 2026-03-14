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
            An *async* callable (``async def fn() -> Any``) that performs
            the actual work.  It will be called with no arguments.

        Raises
        ------
        DuplicateTaskError:
            If a task with the same ``id`` has already been added.
        """
        if task.id in self._tasks:
            raise DuplicateTaskError(f"Task '{task.id}' is already registered.")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_fn

    # ------------------------------------------------------------------
    # Dependency / graph utilities
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all declared dependencies reference registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise MissingDependencyError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'."
                    )

    def _topological_sort(self) -> List[List[str]]:
        """Return tasks grouped into execution waves via Kahn's algorithm.

        Each group in the returned list contains task IDs that can run
        concurrently (all their dependencies are in earlier groups).

        Raises
        ------
        CircularDependencyError:
            If the dependency graph contains a cycle.
        """
        self._validate_dependencies()

        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        # Start with all tasks that have no dependencies
        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: List[List[str]] = []
        visited = 0

        while queue:
            # All tasks currently at zero in-degree form one execution group
            group = list(queue)
            queue.clear()
            groups.append(group)
            visited += len(group)

            next_wave: List[str] = []
            for tid in group:
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_wave.append(dependent)
            queue.extend(next_wave)

        if visited != len(self._tasks):
            involved = [tid for tid, deg in in_degree.items() if deg > 0]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {involved}"
            )

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups for the registered tasks.

        Each inner list contains task IDs that can run concurrently.
        Groups are ordered so that all dependencies of any task appear in
        an earlier group.

        Returns
        -------
        List[List[str]]
            Execution groups sorted by wave order.  Within each group,
            task IDs are sorted by *descending* priority (highest first).

        Raises
        ------
        CircularDependencyError:
            If the dependency graph contains a cycle.
        """
        groups = self._topological_sort()
        # Within each wave sort by descending priority
        return [
            sorted(group, key=lambda tid: self._tasks[tid].priority, reverse=True)
            for group in groups
        ]

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute *task* under *semaphore*, applying retry + backoff logic."""
        metrics = self.metrics.tasks.setdefault(task.id, TaskMetrics(task_id=task.id))

        async with semaphore:
            task.status = TaskStatus.RUNNING
            metrics.start_time = time.monotonic()
            await self.events.emit("on_task_start", task)

            last_exc: Optional[BaseException] = None

            for attempt in range(task.max_retries + 1):
                try:
                    result = await self._coroutines[task.id]()
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    await self.events.emit("on_task_complete", task)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < task.max_retries:
                        task.retry_count += 1
                        backoff = self.base_backoff * (2 ** (attempt))
                        await asyncio.sleep(backoff)
                    else:
                        break

            # All retries exhausted
            task.status = TaskStatus.FAILED
            metrics.end_time = time.monotonic()
            metrics.retry_count = task.retry_count
            await self.events.emit("on_task_fail", task)
            # Re-raise the last exception so callers know the task failed
            if last_exc is not None:
                raise last_exc

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed wave-by-wave.  Within each wave, up to
        ``max_concurrent`` tasks run in parallel.  If any task in a wave
        fails (after exhausting retries), the exception is collected but
        execution continues for the remaining tasks in that wave and
        subsequent waves (fail-fast per-task, not per-wave).

        Returns
        -------
        Dict[str, Any]
            Mapping of task ID to its ``result`` value (``None`` for
            failed tasks).

        Raises
        ------
        CircularDependencyError:
            Propagated from :meth:`get_execution_plan` if a cycle exists.
        ExceptionGroup (Python 3.11+) / gathered exceptions:
            Any task failures are re-raised after all waves complete.
        """
        plan = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        self.metrics.total_start = time.monotonic()
        errors: List[BaseException] = []

        for group in plan:
            coros = [self._run_task(self._tasks[tid], semaphore) for tid in group]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    errors.append(res)

        self.metrics.total_end = time.monotonic()

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("One or more tasks failed", errors)  # type: ignore[name-defined]

        return {tid: task.result for tid, task in self._tasks.items()}
