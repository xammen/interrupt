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
    """Possible states for a task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        self.cycle = cycle or []
        if cycle:
            super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")
        else:
            super().__init__("Circular dependency detected in task graph")


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of retries attempted so far.
        max_retries: Maximum number of retries before marking as permanently failed.
        created_at: Timestamp when the task was created.
        result: The return value of the task coroutine, if completed.
    """

    id: str
    name: str
    priority: int = 5
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
    """Execution metrics collected during a scheduler run.

    Attributes:
        total_time: Wall-clock time for the entire execution in seconds.
        per_task_time: Mapping of task ID to its execution duration in seconds.
        retry_counts: Mapping of task ID to how many retries it needed.
    """

    total_time: float = 0.0
    per_task_time: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)


# Type alias for the coroutine factory that the scheduler invokes per task.
TaskCoroutine = Callable[[Task], Coroutine[Any, Any, Any]]

# Observer callback types.
OnTaskStart = Callable[[Task], None]
OnTaskComplete = Callable[[Task], None]
OnTaskFail = Callable[[Task, Exception], None]


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Features:
        - Priority-based ordering within each dependency level.
        - Topological sort for dependency resolution with cycle detection.
        - Configurable maximum concurrency.
        - Exponential backoff retries for transient failures.
        - Observer pattern for lifecycle events.
        - Execution metrics tracking.

    Args:
        max_concurrency: Maximum number of tasks that can run in parallel.
        base_backoff: Base delay in seconds for exponential backoff (delay = base * 2^retry).
    """

    def __init__(self, max_concurrency: int = 4, base_backoff: float = 1.0) -> None:
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, TaskCoroutine] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff
        self._metrics = TaskMetrics()

        # Observer callbacks.
        self._on_task_start: list[OnTaskStart] = []
        self._on_task_complete: list[OnTaskComplete] = []
        self._on_task_fail: list[OnTaskFail] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coro: TaskCoroutine) -> None:
        """Register a task and its associated coroutine factory.

        Args:
            task: The Task instance to schedule.
            coro: An async callable ``async def fn(task) -> result``.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    def on_task_start(self, callback: OnTaskStart) -> None:
        """Register a callback invoked when a task begins execution."""
        self._on_task_start.append(callback)

    def on_task_complete(self, callback: OnTaskComplete) -> None:
        """Register a callback invoked when a task completes successfully."""
        self._on_task_complete.append(callback)

    def on_task_fail(self, callback: OnTaskFail) -> None:
        """Register a callback invoked when a task fails (after all retries exhausted)."""
        self._on_task_fail.append(callback)

    def get_execution_plan(self) -> list[list[str]]:
        """Return ordered groups of task IDs representing the execution plan.

        Tasks within the same group are independent and can run concurrently.
        Groups are ordered so that all dependencies of group *n* appear in
        groups *0 .. n-1*.

        Returns:
            A list of lists of task IDs.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        return self._topological_groups()

    async def run(self) -> TaskMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A ``TaskMetrics`` instance with timing and retry information.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._validate_dependencies()
        groups = self._topological_groups()
        self._metrics = TaskMetrics()

        overall_start = time.monotonic()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in groups:
            # Sort by priority descending within each group.
            sorted_ids = sorted(group, key=lambda tid: self._tasks[tid].priority, reverse=True)
            await asyncio.gather(
                *(self._run_task(tid, semaphore) for tid in sorted_ids)
            )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    @property
    def metrics(self) -> TaskMetrics:
        """Access the most recent execution metrics."""
        return self._metrics

    @property
    def tasks(self) -> dict[str, Task]:
        """Access registered tasks by ID."""
        return dict(self._tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all dependency references exist and there are no cycles.

        Raises:
            ValueError: If a dependency references an unknown task.
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )
        self._detect_cycle()

    def _detect_cycle(self) -> None:
        """Detect cycles using DFS with three-colour marking.

        Raises:
            CircularDependencyError: On the first cycle found.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(tid: str) -> None:
            colour[tid] = GRAY
            for dep_id in self._tasks[tid].dependencies:
                if colour[dep_id] == GRAY:
                    # Reconstruct the cycle.
                    cycle: list[str] = [dep_id, tid]
                    node: str = tid
                    p = parent[node]
                    while p is not None and p != dep_id:
                        cycle.append(p)
                        node = p
                        p = parent[node]
                    cycle.append(dep_id)
                    cycle.reverse()
                    raise CircularDependencyError(cycle)
                if colour[dep_id] == WHITE:
                    parent[dep_id] = tid
                    dfs(dep_id)
            colour[tid] = BLACK

        for tid in self._tasks:
            if colour[tid] == WHITE:
                dfs(tid)

    def _topological_groups(self) -> list[list[str]]:
        """Kahn's algorithm producing grouped layers for parallel execution.

        Returns:
            Ordered list of groups. Each group contains task IDs whose
            dependencies have all appeared in earlier groups.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        groups: list[list[str]] = []

        while queue:
            current_group = list(queue)
            queue.clear()
            groups.append(current_group)
            for tid in current_group:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        queue.append(dep_tid)

        return groups

    async def _run_task(self, task_id: str, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + backoff under the semaphore."""
        task = self._tasks[task_id]
        coro_factory = self._coroutines[task_id]

        async with semaphore:
            while True:
                task.status = TaskStatus.RUNNING
                for cb in self._on_task_start:
                    cb(task)

                task_start = time.monotonic()
                try:
                    task.result = await coro_factory(task)
                    elapsed = time.monotonic() - task_start
                    task.status = TaskStatus.COMPLETED
                    self._metrics.per_task_time[task.id] = elapsed
                    self._metrics.retry_counts[task.id] = task.retry_count
                    for cb in self._on_task_complete:
                        cb(task)
                    return
                except Exception as exc:
                    elapsed = time.monotonic() - task_start
                    task.retry_count += 1
                    if task.retry_count <= task.max_retries:
                        delay = self._base_backoff * (2 ** (task.retry_count - 1))
                        await asyncio.sleep(delay)
                        # Loop back and retry.
                    else:
                        task.status = TaskStatus.FAILED
                        self._metrics.per_task_time[task.id] = elapsed
                        self._metrics.retry_counts[task.id] = task.retry_count
                        for cb in self._on_task_fail:
                            cb(task, exc)
                        return
