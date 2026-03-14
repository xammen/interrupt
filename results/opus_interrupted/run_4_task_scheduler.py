"""
Async task scheduler with priority queue, dependency resolution,
retry logic, concurrency control, and observer-pattern event emission.
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
    """Raised when the dependency graph contains a cycle."""


class TaskNotFoundError(Exception):
    """Raised when a referenced task ID does not exist."""


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work to be scheduled.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: IDs of tasks that must complete before this one.
        status: Current lifecycle status.
        retry_count: How many times the task has been retried so far.
        max_retries: Maximum number of retries before permanent failure.
        created_at: Timestamp of task creation.
        result: The value returned by the task coroutine, if any.
    """
    id: str
    name: str
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=__import__("datetime").timezone.utc))
    result: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Timing and retry information for a single task execution."""
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    retries: int = 0
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class SchedulerMetrics:
    """Aggregate metrics for the entire scheduler run."""
    total_time: float = 0.0
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)

    @property
    def total_retries(self) -> int:
        return sum(m.retries for m in self.task_metrics.values())


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EventCallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
        base_backoff: Base delay (seconds) for exponential-backoff retries.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        base_backoff: float = 0.1,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff

        # Observer-pattern listeners
        self._listeners: Dict[str, List[EventCallback]] = defaultdict(list)

        # Populated after execution
        self.metrics = SchedulerMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The Task descriptor.
            coro: An async callable that performs the actual work.
                  It receives the Task instance as its sole argument.
        """
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    # -- Observer helpers ----------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> None:
        """Subscribe *callback* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.
        """
        self._listeners[event].append(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for cb in self._listeners.get(event, []):
            cb(*args, **kwargs)

    # -- Dependency resolution ------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency reference points to a known task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise TaskNotFoundError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_circular_dependencies(self) -> None:
        """Raise :class:`CircularDependencyError` if cycles exist.

        Uses iterative DFS with WHITE/GRAY/BLACK colouring.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._tasks}

        for start in self._tasks:
            if color[start] != WHITE:
                continue
            stack: List[tuple[str, int]] = [(start, 0)]
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx == 0:
                    if color[node] == GRAY:
                        raise CircularDependencyError(
                            f"Circular dependency detected involving task '{node}'"
                        )
                    if color[node] == BLACK:
                        continue
                    color[node] = GRAY
                if idx < len(deps):
                    stack.append((node, idx + 1))
                    child = deps[idx]
                    if color[child] == GRAY:
                        raise CircularDependencyError(
                            f"Circular dependency detected involving task '{child}'"
                        )
                    if color[child] == WHITE:
                        stack.append((child, 0))
                else:
                    color[node] = BLACK

    def _topological_groups(self) -> List[List[str]]:
        """Return tasks grouped by execution wave (Kahn's algorithm).

        Tasks within the same group are independent and may run concurrently.
        Higher-priority tasks appear earlier inside each group.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Seed with tasks that have no unmet dependencies
        ready: List[str] = [
            tid for tid, deg in in_degree.items() if deg == 0
        ]

        groups: List[List[str]] = []
        while ready:
            # Sort current wave by descending priority
            ready.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(list(ready))
            next_ready: List[str] = []
            for tid in ready:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_ready.append(dep_tid)
            ready = next_ready

        return groups

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        Raises :class:`CircularDependencyError` if cycles exist.

        Returns:
            A list of lists of task IDs.
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()
        return self._topological_groups()

    # -- Execution ------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + exponential backoff."""
        tm = TaskMetrics(task_id=task.id)
        self.metrics.task_metrics[task.id] = tm

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                self._emit("on_task_start", task)
                tm.start_time = time.monotonic()

                try:
                    coro_fn = self._coroutines[task.id]
                    task.result = await coro_fn(task)
                    tm.end_time = time.monotonic()
                    tm.duration = tm.end_time - tm.start_time
                    task.status = TaskStatus.COMPLETED
                    tm.status = TaskStatus.COMPLETED
                    tm.retries = task.retry_count
                    self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    tm.end_time = time.monotonic()
                    tm.duration = tm.end_time - tm.start_time
                    task.retry_count += 1

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        tm.status = TaskStatus.FAILED
                        tm.retries = task.retry_count
                        self._emit("on_task_fail", task, exc)
                        return

                    # Exponential backoff
                    delay = self._base_backoff * (2 ** (task.retry_count - 1))
                    await asyncio.sleep(delay)

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A :class:`SchedulerMetrics` instance with timing data.

        Raises:
            CircularDependencyError: If the dependency graph has cycles.
            TaskNotFoundError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()

        groups = self._topological_groups()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()

        for group in groups:
            # Before launching, propagate any upstream failures
            failed_ids: Set[str] = {
                t.id for t in self._tasks.values()
                if t.status == TaskStatus.FAILED
            }
            if failed_ids:
                self._propagate_failures(failed_ids)

            # Only run tasks that are still PENDING
            tasks_in_group = [
                self._tasks[tid] for tid in group
                if self._tasks[tid].status == TaskStatus.PENDING
            ]
            if tasks_in_group:
                await asyncio.gather(
                    *(self._run_task(t, semaphore) for t in tasks_in_group)
                )

        self.metrics.total_time = time.monotonic() - overall_start
        return self.metrics

    def _propagate_failures(self, failed_ids: Set[str]) -> None:
        """Mark tasks whose dependencies have failed as FAILED."""
        changed = True
        while changed:
            changed = False
            for task in self._tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                if any(d in failed_ids for d in task.dependencies):
                    task.status = TaskStatus.FAILED
                    failed_ids.add(task.id)
                    tm = TaskMetrics(
                        task_id=task.id, status=TaskStatus.FAILED,
                    )
                    self.metrics.task_metrics[task.id] = tm
                    self._emit("on_task_fail", task, RuntimeError(
                        f"Skipped: dependency failed"
                    ))
                    changed = True
