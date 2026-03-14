"""
Async task scheduler with dependency resolution, retry logic, and concurrency control.

Provides a priority-based task scheduler that resolves dependency graphs via
topological sort, runs independent tasks concurrently with configurable limits,
and implements exponential backoff retry logic for failed tasks.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class TaskStatus(Enum):
    """Status of a task in the scheduler."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        self.cycle = cycle or []
        if self.cycle:
            cycle_str = " -> ".join(self.cycle)
            super().__init__(f"Circular dependency detected: {cycle_str}")
        else:
            super().__init__("Circular dependency detected in task graph")


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable task name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of retries attempted so far.
        max_retries: Maximum number of retry attempts before marking as failed.
        created_at: Timestamp when the task was created.
        result: The return value of the task's coroutine, if completed.
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
    """Execution metrics for a single task.

    Attributes:
        task_id: The ID of the task these metrics belong to.
        start_time: When execution started (seconds since epoch).
        end_time: When execution ended (seconds since epoch).
        duration: Total wall-clock execution time in seconds.
        retry_count: How many retries were needed.
        status: Final status of the task.
    """

    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for a scheduler run.

    Attributes:
        total_time: Wall-clock time for the entire scheduler run.
        task_metrics: Per-task metrics keyed by task ID.
        total_retries: Sum of all retries across all tasks.
        completed_count: Number of tasks that completed successfully.
        failed_count: Number of tasks that failed after exhausting retries.
    """

    total_time: float = 0.0
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)
    total_retries: int = 0
    completed_count: int = 0
    failed_count: int = 0


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Maintains a registry of tasks, resolves their dependency graph via
    topological sort, and executes them concurrently (up to a configurable
    limit) with exponential-backoff retry logic on failure.

    Args:
        max_concurrency: Maximum number of tasks to run in parallel.
        base_retry_delay: Base delay in seconds for exponential backoff.

    Example::

        scheduler = TaskScheduler(max_concurrency=3)

        async def my_work() -> str:
            return "done"

        scheduler.add_task(Task(id="t1", name="Task 1"), my_work)
        scheduler.on_task_complete(lambda t: print(f"{t.name} finished"))
        metrics = await scheduler.run()
    """

    def __init__(
        self, max_concurrency: int = 4, base_retry_delay: float = 1.0
    ) -> None:
        self._tasks: dict[str, Task] = {}
        self._coroutine_factories: dict[
            str, Callable[[], Coroutine[Any, Any, Any]]
        ] = {}
        self._max_concurrency = max_concurrency
        self._base_retry_delay = base_retry_delay
        self._metrics = SchedulerMetrics()

        # Observer pattern: event -> list of callbacks
        self._listeners: dict[str, list[Callable[[Task], Any]]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # Task management
    # ------------------------------------------------------------------ #

    def add_task(
        self,
        task: Task,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine factory.

        Args:
            task: The Task instance to schedule.
            coroutine_factory: A zero-argument callable that returns a new
                coroutine each time it is called (needed for retries).

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutine_factories[task.id] = coroutine_factory

    def get_task(self, task_id: str) -> Task:
        """Retrieve a task by its ID.

        Raises:
            KeyError: If no task with the given ID exists.
        """
        return self._tasks[task_id]

    # ------------------------------------------------------------------ #
    # Observer pattern
    # ------------------------------------------------------------------ #

    def on_task_start(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task begins execution."""
        self._listeners["on_task_start"].append(callback)

    def on_task_complete(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task completes successfully."""
        self._listeners["on_task_complete"].append(callback)

    def on_task_fail(self, callback: Callable[[Task], Any]) -> None:
        """Register a callback invoked when a task fails (after all retries)."""
        self._listeners["on_task_fail"].append(callback)

    def _emit(self, event: str, task: Task) -> None:
        """Fire all registered callbacks for *event*."""
        for cb in self._listeners.get(event, []):
            cb(task)

    # ------------------------------------------------------------------ #
    # Dependency resolution
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from registered tasks.

        Returns:
            A tuple of (adjacency list, in-degree map).

        Raises:
            ValueError: If a dependency references an unregistered task.
        """
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )
                adj[dep_id].append(task.id)
                in_degree[task.id] += 1

        return adj, in_degree

    def _detect_circular_dependencies(self) -> None:
        """Raise CircularDependencyError if the task graph contains a cycle.

        Uses DFS-based cycle detection to identify the actual cycle path.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for dep_id in self._tasks[node].dependencies:
                if dep_id not in self._tasks:
                    continue
                if color[dep_id] == GRAY:
                    # Found a cycle – reconstruct it
                    cycle = [dep_id, node]
                    cur = node
                    while cur != dep_id:
                        cur = parent[cur]  # type: ignore[assignment]
                        if cur is None:
                            break
                        cycle.append(cur)
                    cycle.reverse()
                    return cycle
                if color[dep_id] == WHITE:
                    parent[dep_id] = node
                    result = dfs(dep_id)
                    if result is not None:
                        return result
            color[node] = BLACK
            return None

        for tid in self._tasks:
            if color[tid] == WHITE:
                cycle = dfs(tid)
                if cycle is not None:
                    raise CircularDependencyError(cycle)

    def _topological_sort_groups(self) -> list[list[str]]:
        """Return tasks grouped by execution layer (Kahn's algorithm).

        Each inner list contains tasks that can run concurrently because all
        of their dependencies belong to earlier groups.

        Returns:
            Ordered list of execution groups (each group is a list of task IDs
            sorted by descending priority).

        Raises:
            CircularDependencyError: If a cycle is detected.
        """
        self._detect_circular_dependencies()

        adj, in_degree = self._build_graph()
        queue: deque[str] = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)

        groups: list[list[str]] = []
        visited = 0

        while queue:
            # All items currently in the queue form one execution group
            group: list[str] = []
            for _ in range(len(queue)):
                tid = queue.popleft()
                group.append(tid)
                visited += 1
                for neighbor in adj[tid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # Sort group by priority descending so higher-priority tasks start first
            group.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(group)

        if visited != len(self._tasks):
            raise CircularDependencyError()

        return groups

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        Groups are ordered so that every task's dependencies appear in an
        earlier group.

        Returns:
            List of execution groups.

        Raises:
            CircularDependencyError: If a cycle exists in the dependency graph.
        """
        return self._topological_sort_groups()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def _execute_task(self, task_id: str, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry logic and exponential backoff.

        Args:
            task_id: ID of the task to execute.
            semaphore: Semaphore enforcing the concurrency limit.
        """
        task = self._tasks[task_id]
        metrics = TaskMetrics(task_id=task_id)
        metrics.start_time = time.monotonic()

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                self._emit("on_task_start", task)

                try:
                    coro = self._coroutine_factories[task_id]()
                    task.result = await coro
                    task.status = TaskStatus.COMPLETED
                    metrics.end_time = time.monotonic()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.retry_count = task.retry_count
                    metrics.status = TaskStatus.COMPLETED
                    self._metrics.task_metrics[task_id] = metrics
                    self._metrics.completed_count += 1
                    self._emit("on_task_complete", task)
                    return
                except Exception:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        metrics.end_time = time.monotonic()
                        metrics.duration = metrics.end_time - metrics.start_time
                        metrics.retry_count = task.retry_count
                        metrics.status = TaskStatus.FAILED
                        self._metrics.task_metrics[task_id] = metrics
                        self._metrics.failed_count += 1
                        self._emit("on_task_fail", task)
                        return

            # Exponential backoff (outside semaphore so we don't hold the slot)
            delay = self._base_retry_delay * (2 ** (task.retry_count - 1))
            await asyncio.sleep(delay)

    async def run(self) -> SchedulerMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            SchedulerMetrics with per-task and aggregate timing information.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            ValueError: If a dependency references an unknown task.
        """
        self._metrics = SchedulerMetrics()
        overall_start = time.monotonic()

        groups = self._topological_sort_groups()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for group in groups:
            # Check that all dependencies completed; skip tasks whose deps failed
            runnable: list[str] = []
            for tid in group:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_ok:
                    runnable.append(tid)
                else:
                    task.status = TaskStatus.FAILED
                    task_metrics = TaskMetrics(
                        task_id=tid,
                        status=TaskStatus.FAILED,
                    )
                    self._metrics.task_metrics[tid] = task_metrics
                    self._metrics.failed_count += 1
                    self._emit("on_task_fail", task)

            # Run all tasks in this group concurrently (bounded by semaphore)
            await asyncio.gather(
                *(self._execute_task(tid, semaphore) for tid in runnable)
            )

        overall_end = time.monotonic()
        self._metrics.total_time = overall_end - overall_start
        self._metrics.total_retries = sum(
            m.retry_count for m in self._metrics.task_metrics.values()
        )
        return self._metrics

    @property
    def metrics(self) -> SchedulerMetrics:
        """Return the metrics from the most recent run."""
        return self._metrics
