"""
Async task scheduler with priority queuing, dependency resolution,
retry logic, concurrency control, and an observer-based event system.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        if cycle:
            msg = f"Circular dependency detected: {' -> '.join(cycle)}"
        else:
            msg = "Circular dependency detected in the task graph"
        super().__init__(msg)
        self.cycle = cycle


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Task:
    """Represents a unit of work inside the scheduler.

    Attributes:
        id:           Unique identifier for the task.
        name:         Human-readable task name.
        priority:     Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task.
        status:       Current lifecycle status.
        retry_count:  How many times the task has been retried so far.
        max_retries:  Maximum number of retry attempts before final failure.
        created_at:   Timestamp of task creation.
        result:       The return value of the task coroutine (set after completion).
    """
    id: str
    name: str
    priority: int = 5
    dependencies: list[str] = field(default_factory=list)
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
    """Timing and retry information for a single task execution."""
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    retries: int = 0

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds the task took (including retries)."""
        return self.end_time - self.start_time


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for a full scheduler run."""
    total_time: float = 0.0
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observer / event types
# ---------------------------------------------------------------------------

# Callback signatures accepted by the observer system.
EventCallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Features:
        * Priority-aware execution within each dependency tier.
        * Topological-sort-based dependency resolution.
        * Circular-dependency detection.
        * Configurable concurrency limit.
        * Exponential-backoff retry on failure.
        * Observer-pattern event hooks (on_task_start, on_task_complete,
          on_task_fail).

    Example::

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(Task(id="a", name="Step A"), coro=my_coroutine)
        plan = scheduler.get_execution_plan()
        metrics = await scheduler.run()
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max_concurrency = max_concurrency
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._metrics = SchedulerMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(
        self,
        task: Task,
        coro: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The :class:`Task` descriptor.
            coro: An async callable that will be awaited when the task runs.
                  It receives the :class:`Task` instance as its sole argument.

        Raises:
            ValueError: If a task with the same *id* is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    # -- Observer pattern -----------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> None:
        """Subscribe *callback* to *event*.

        Supported events: ``on_task_start``, ``on_task_complete``,
        ``on_task_fail``.
        """
        self._listeners[event].append(callback)

    async def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all callbacks registered for *event*."""
        for cb in self._listeners.get(event, []):
            ret = cb(*args, **kwargs)
            if asyncio.iscoroutine(ret):
                await ret

    # -- Dependency graph helpers ---------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every dependency references a registered task."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _detect_circular_dependencies(self) -> None:
        """Raise :class:`CircularDependencyError` if the graph has a cycle.

        Uses iterative DFS with three-colour marking (WHITE / GREY / BLACK).
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        for start in self._tasks:
            if colour[start] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, idx = stack.pop()
                deps = self._tasks[node].dependencies
                if idx == 0:
                    colour[node] = GREY
                if idx < len(deps):
                    stack.append((node, idx + 1))
                    dep = deps[idx]
                    if colour[dep] == GREY:
                        # Reconstruct cycle path
                        cycle = [dep, node]
                        p = parent.get(node)
                        while p is not None and p != dep:
                            cycle.append(p)
                            p = parent.get(p)
                        cycle.append(dep)
                        cycle.reverse()
                        raise CircularDependencyError(cycle)
                    if colour[dep] == WHITE:
                        parent[dep] = node
                        stack.append((dep, 0))
                else:
                    colour[node] = BLACK

    def _topological_groups(self) -> list[list[Task]]:
        """Return tasks grouped by dependency tiers (Kahn's algorithm).

        Each inner list contains tasks that can run concurrently.  Within each
        group the tasks are sorted by *priority* (highest first).

        Raises:
            CircularDependencyError: If not all tasks can be scheduled.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue: list[str] = [tid for tid, d in in_degree.items() if d == 0]
        groups: list[list[Task]] = []
        visited = 0

        while queue:
            # Sort the current frontier by priority (descending)
            group = sorted(
                [self._tasks[tid] for tid in queue],
                key=lambda t: t.priority,
                reverse=True,
            )
            groups.append(group)
            visited += len(group)
            next_queue: list[str] = []
            for task in group:
                for dep_tid in dependents[task.id]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_queue.append(dep_tid)
            queue = next_queue

        if visited != len(self._tasks):
            raise CircularDependencyError()

        return groups

    # -- Execution plan -------------------------------------------------------

    def get_execution_plan(self) -> list[list[Task]]:
        """Return the ordered execution groups without running anything.

        Each group is a list of :class:`Task` objects that may run in
        parallel.  Groups must be executed sequentially (a group's tasks
        depend on all prior groups having completed).

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            ValueError: If a dependency references an unknown task.
        """
        self._validate_dependencies()
        self._detect_circular_dependencies()
        return self._topological_groups()

    # -- Single-task runner with retries --------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task, retrying with exponential backoff on failure."""
        metrics = TaskMetrics(task_id=task.id)
        metrics.start_time = time.monotonic()

        async with semaphore:
            while True:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)
                try:
                    result = await self._coroutines[task.id](task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.retries = task.retry_count
                    self._metrics.task_metrics[task.id] = metrics
                    await self._emit("on_task_complete", task)
                    return
                except Exception as exc:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.result = exc
                        metrics.end_time = time.monotonic()
                        metrics.retries = task.retry_count
                        self._metrics.task_metrics[task.id] = metrics
                        await self._emit("on_task_fail", task, exc)
                        return
                    # Exponential backoff: 2^(retry-1) seconds, e.g. 1, 2, 4 …
                    backoff = 2 ** (task.retry_count - 1)
                    await asyncio.sleep(backoff)

    # -- Main entry point -----------------------------------------------------

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A :class:`SchedulerMetrics` instance with timing information.

        Raises:
            CircularDependencyError: If a cycle exists in the dependency graph.
            ValueError: If a dependency references an unknown task.
        """
        groups = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics = SchedulerMetrics()

        overall_start = time.monotonic()

        for group in groups:
            # Check that all dependencies completed successfully
            runnable: list[Task] = []
            for task in group:
                failed_deps = [
                    d for d in task.dependencies
                    if self._tasks[d].status != TaskStatus.COMPLETED
                ]
                if failed_deps:
                    task.status = TaskStatus.FAILED
                    task.result = RuntimeError(
                        f"Dependency tasks failed: {failed_deps}"
                    )
                    await self._emit("on_task_fail", task, task.result)
                else:
                    runnable.append(task)

            await asyncio.gather(
                *(self._run_task(t, semaphore) for t in runnable)
            )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    # -- Accessors ------------------------------------------------------------

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` registered under *task_id*.

        Raises:
            KeyError: If *task_id* is not registered.
        """
        return self._tasks[task_id]

    @property
    def tasks(self) -> dict[str, Task]:
        """Read-only view of all registered tasks."""
        return dict(self._tasks)

    @property
    def metrics(self) -> SchedulerMetrics:
        """Metrics from the most recent :meth:`run` invocation."""
        return self._metrics
