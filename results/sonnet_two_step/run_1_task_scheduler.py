"""
Async task scheduler with priority queue, dependency resolution,
concurrent execution, retry logic, and observer pattern.
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
# Exceptions
# ---------------------------------------------------------------------------


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected among tasks."""


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist in the scheduler."""


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work managed by the scheduler.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum allowed retries before marking as FAILED.
        created_at: Timestamp when the task was created.
        result: The return value of the task coroutine, if completed.
    """

    id: str
    name: str
    priority: int  # 1–10
    dependencies: List[str] = field(default_factory=list)
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
    retry_count: int = 0

    @property
    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds spent executing (excluding retries wait time)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


# ---------------------------------------------------------------------------
# Observer / event types
# ---------------------------------------------------------------------------

EventCallback = Callable[["Task"], Coroutine[Any, Any, None]]


class _EventBus:
    """Simple async observer bus supporting multiple listeners per event."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event: str, callback: EventCallback) -> None:
        self._listeners[event].append(callback)

    async def emit(self, event: str, task: "Task") -> None:
        for cb in self._listeners[event]:
            await cb(task)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Async task scheduler with dependency resolution and observer support.

    Features
    --------
    * Priority queue — higher-priority tasks (closer to 10) run first.
    * Topological sort — dependency graph is resolved before execution.
    * Circular dependency detection — raises :class:`CircularDependencyError`.
    * Concurrent execution — independent tasks run in parallel up to
      *max_concurrency* at a time.
    * Exponential back-off retry — failed tasks are retried with increasing
      delays (base 2 ** attempt seconds, capped at 60 s).
    * Observer pattern — subscribe async callbacks to ``on_task_start``,
      ``on_task_complete``, and ``on_task_fail`` events.
    * Execution plan — :meth:`get_execution_plan` returns ordered groups of
      tasks that can run concurrently.
    * Metrics — per-task timing, retry counts, and total scheduler wall time.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks allowed to run in parallel (default 4).
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._max_concurrency = max_concurrency
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {}
        self._bus = _EventBus()
        self._metrics: Dict[str, TaskMetrics] = {}
        self._total_start: Optional[float] = None
        self._total_end: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API – registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task: Task,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine factory.

        Parameters
        ----------
        task:
            The :class:`Task` metadata object.
        coroutine_factory:
            A zero-argument callable that returns a *fresh* coroutine each
            time it is called (necessary for retries).
        """
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_factory
        self._metrics[task.id] = TaskMetrics(task_id=task.id)

    def on_task_start(self, callback: EventCallback) -> None:
        """Subscribe an async callback fired just before a task begins."""
        self._bus.subscribe("on_task_start", callback)

    def on_task_complete(self, callback: EventCallback) -> None:
        """Subscribe an async callback fired when a task succeeds."""
        self._bus.subscribe("on_task_complete", callback)

    def on_task_fail(self, callback: EventCallback) -> None:
        """Subscribe an async callback fired when a task exhausts its retries."""
        self._bus.subscribe("on_task_fail", callback)

    # ------------------------------------------------------------------
    # Public API – inspection
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups derived from the dependency graph.

        Each group contains task IDs that may run concurrently (all their
        dependencies belong to earlier groups).  Groups are ordered so that
        earlier groups must complete before the next begins.

        Returns
        -------
        list[list[str]]
            Ordered list of concurrent task-ID groups.

        Raises
        ------
        CircularDependencyError
            If a cycle is detected in the dependency graph.
        TaskNotFoundError
            If a dependency references an unknown task ID.
        """
        self._validate_dependencies()
        return self._build_level_groups()

    @property
    def metrics(self) -> Dict[str, TaskMetrics]:
        """Per-task :class:`TaskMetrics` keyed by task ID."""
        return dict(self._metrics)

    @property
    def total_elapsed(self) -> Optional[float]:
        """Total wall-clock seconds for the last :meth:`run` call."""
        if self._total_start is not None and self._total_end is not None:
            return self._total_end - self._total_start
        return None

    # ------------------------------------------------------------------
    # Public API – execution
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute all registered tasks respecting dependencies and priority.

        Tasks are executed level-by-level (topological layers).  Within each
        level, tasks are sorted by descending priority and run concurrently up
        to *max_concurrency*.

        Returns
        -------
        dict[str, Any]
            Mapping of task ID to its result (or ``None`` if failed/no result).

        Raises
        ------
        CircularDependencyError
            If a cycle is detected before execution begins.
        TaskNotFoundError
            If a dependency references an unknown task ID.
        """
        self._total_start = time.monotonic()
        groups = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in groups:
            # Sort descending by priority so highest-priority tasks are
            # scheduled first within the asyncio event loop.
            sorted_group = sorted(
                group,
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
            await asyncio.gather(
                *(self._execute_task(tid, semaphore) for tid in sorted_group)
            )

        self._total_end = time.monotonic()
        return {tid: self._tasks[tid].result for tid in self._tasks}

    # ------------------------------------------------------------------
    # Internal helpers – dependency graph
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Check that all dependency IDs exist in the scheduler."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list: task_id -> [tasks that depend on it]."""
        adj: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                adj[dep_id].append(task.id)
        return adj

    def _build_in_degree(self) -> Dict[str, int]:
        return {
            tid: len(task.dependencies) for tid, task in self._tasks.items()
        }

    def _build_level_groups(self) -> List[List[str]]:
        """Kahn's algorithm — returns layered groups, raises on cycle."""
        in_degree = self._build_in_degree()
        adj = self._build_adjacency()

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: List[List[str]] = []
        visited = 0

        while queue:
            # Collect all zero-in-degree nodes as one concurrent group
            level_size = len(queue)
            group: List[str] = []
            for _ in range(level_size):
                tid = queue.popleft()
                group.append(tid)
                visited += 1
                for neighbour in adj[tid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        queue.append(neighbour)
            groups.append(group)

        if visited != len(self._tasks):
            cycle_nodes = [
                tid for tid, deg in in_degree.items() if deg > 0
            ]
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {cycle_nodes}"
            )

        return groups

    # ------------------------------------------------------------------
    # Internal helpers – execution & retry
    # ------------------------------------------------------------------

    async def _execute_task(
        self, task_id: str, semaphore: asyncio.Semaphore
    ) -> None:
        """Acquire the semaphore then run the task with retry/back-off."""
        async with semaphore:
            task = self._tasks[task_id]
            metrics = self._metrics[task_id]

            attempt = 0
            while True:
                task.status = TaskStatus.RUNNING
                metrics.start_time = time.monotonic()
                await self._bus.emit("on_task_start", task)

                try:
                    coro = self._coroutines[task_id]()
                    task.result = await coro
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retry_count = task.retry_count
                    await self._bus.emit("on_task_complete", task)
                    return

                except Exception as exc:  # noqa: BLE001
                    metrics.end_time = time.monotonic()
                    attempt += 1
                    task.retry_count += 1

                    if task.retry_count <= task.max_retries:
                        delay = min(2 ** attempt, 60)
                        await asyncio.sleep(delay)
                        # Reset start time for the next attempt
                        metrics.start_time = time.monotonic()
                        continue

                    # Exhausted retries
                    task.status = TaskStatus.FAILED
                    metrics.retry_count = task.retry_count
                    await self._bus.emit("on_task_fail", task)
                    return
