"""Async task scheduler with dependency resolution, retry logic, and concurrency control."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
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

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum number of retry attempts before permanent failure.
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
        task_id: The task this metric belongs to.
        start_time: When execution began (epoch seconds).
        end_time: When execution finished (epoch seconds).
        duration: Wall-clock duration in seconds.
        retries: How many retries were needed.
    """

    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    retries: int = 0


@dataclass
class SchedulerMetrics:
    """Aggregate execution metrics for the entire scheduler run.

    Attributes:
        total_time: Wall-clock time for the full execution run.
        task_metrics: Per-task metrics keyed by task ID.
    """

    total_time: float = 0.0
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)


# Type alias for task coroutine factories.
TaskCoroutine = Callable[..., Coroutine[Any, Any, Any]]

# Type alias for event listener callbacks.
EventListener = Callable[[Task], Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution, retries, and concurrency control.

    Features:
        - Priority-aware execution within each dependency tier.
        - Topological sort for dependency resolution with cycle detection.
        - Configurable concurrency limit.
        - Exponential-backoff retry logic for transient failures.
        - Observer-pattern event emission (on_task_start, on_task_complete, on_task_fail).
        - Execution-plan introspection via ``get_execution_plan``.
        - Per-task and aggregate metrics tracking.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, TaskCoroutine] = {}
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self._max_concurrency = max_concurrency
        self._metrics = SchedulerMetrics()

    # ------------------------------------------------------------------
    # Public API – task registration
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coroutine: TaskCoroutine) -> None:
        """Register a task and its associated coroutine factory.

        Args:
            task: The ``Task`` instance to schedule.
            coroutine: An async callable that will be awaited when the task runs.
                       It receives the ``Task`` as its sole argument.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine

    # ------------------------------------------------------------------
    # Public API – observer pattern
    # ------------------------------------------------------------------

    def on_task_start(self, listener: EventListener) -> None:
        """Register a listener invoked when a task begins execution."""
        self._listeners["on_task_start"].append(listener)

    def on_task_complete(self, listener: EventListener) -> None:
        """Register a listener invoked when a task completes successfully."""
        self._listeners["on_task_complete"].append(listener)

    def on_task_fail(self, listener: EventListener) -> None:
        """Register a listener invoked when a task fails (after all retries)."""
        self._listeners["on_task_fail"].append(listener)

    # ------------------------------------------------------------------
    # Public API – execution plan & metrics
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups (topological layers).

        Each inner list contains task IDs that can run concurrently.  The
        outer list is ordered so that every task's dependencies appear in
        an earlier group.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.

        Returns:
            A list of groups, where each group is a list of task IDs sorted
            by descending priority.
        """
        self._validate_dependencies()
        topo_order = self._topological_sort()
        return self._build_execution_groups(topo_order)

    @property
    def metrics(self) -> SchedulerMetrics:
        """Access the collected execution metrics."""
        return self._metrics

    # ------------------------------------------------------------------
    # Public API – execution
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A mapping of task ID to its result value.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        execution_groups = self.get_execution_plan()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()
        results: dict[str, Any] = {}

        for group in execution_groups:
            tasks_in_group = [
                self._run_task(task_id, semaphore) for task_id in group
            ]
            group_results = await asyncio.gather(*tasks_in_group, return_exceptions=True)
            for task_id, res in zip(group, group_results):
                if isinstance(res, BaseException):
                    results[task_id] = None
                else:
                    results[task_id] = res

        self._metrics.total_time = time.monotonic() - overall_start
        return results

    # ------------------------------------------------------------------
    # Internal – dependency graph helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all referenced dependency IDs actually exist."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological sorting with cycle detection.

        Returns:
            A list of task IDs in valid execution order.

        Raises:
            CircularDependencyError: If a cycle exists.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                adjacency[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue: list[str] = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda tid: -self._tasks[tid].priority,
        )
        order: list[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            # Keep the queue sorted by priority (highest first).
            queue.sort(key=lambda tid: -self._tasks[tid].priority)

        if len(order) != len(self._tasks):
            # Find the cycle for a useful error message.
            cycle = self._find_cycle()
            raise CircularDependencyError(cycle)

        return order

    def _find_cycle(self) -> list[str]:
        """DFS-based cycle detection; returns one cycle path."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, Optional[str]] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> Optional[list[str]]:
            color[node] = GRAY
            for task in self._tasks.values():
                if node in task.dependencies:
                    continue
            # Check outgoing edges (node -> tasks that depend on node is wrong;
            # we need node's own dependencies).
            for dep_id in self._tasks[node].dependencies:
                if color[dep_id] == GRAY:
                    # Reconstruct cycle.
                    cycle = [dep_id, node]
                    cur = parent.get(node)
                    while cur and cur != dep_id:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    cycle.append(dep_id)
                    cycle.reverse()
                    return cycle
                if color[dep_id] == WHITE:
                    parent[dep_id] = node
                    result = dfs(dep_id)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for tid in self._tasks:
            if color[tid] == WHITE:
                result = dfs(tid)
                if result:
                    return result
        return []  # Should not reach here if called after detecting a cycle.

    def _build_execution_groups(self, topo_order: list[str]) -> list[list[str]]:
        """Partition the topological order into concurrent execution tiers.

        A task's tier is ``max(tier(dep) for dep in dependencies) + 1``, or 0
        if it has no dependencies.
        """
        tier: dict[str, int] = {}
        for tid in topo_order:
            task = self._tasks[tid]
            if not task.dependencies:
                tier[tid] = 0
            else:
                tier[tid] = max(tier[dep] for dep in task.dependencies) + 1

        max_tier = max(tier.values()) if tier else 0
        groups: list[list[str]] = [[] for _ in range(max_tier + 1)]
        for tid in topo_order:
            groups[tier[tid]].append(tid)

        # Sort each group by descending priority.
        for group in groups:
            group.sort(key=lambda tid: -self._tasks[tid].priority)

        return groups

    # ------------------------------------------------------------------
    # Internal – task execution
    # ------------------------------------------------------------------

    async def _run_task(self, task_id: str, semaphore: asyncio.Semaphore) -> Any:
        """Execute a single task with retry logic and metrics collection."""
        task = self._tasks[task_id]
        metrics = TaskMetrics(task_id=task_id)

        # Check that all dependencies completed successfully.
        for dep_id in task.dependencies:
            dep = self._tasks[dep_id]
            if dep.status == TaskStatus.FAILED:
                task.status = TaskStatus.FAILED
                self._metrics.task_metrics[task_id] = metrics
                self._emit("on_task_fail", task)
                return None

        while task.retry_count <= task.max_retries:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                self._emit("on_task_start", task)
                metrics.start_time = time.monotonic()

                try:
                    coro = self._coroutines[task_id]
                    result = await coro(task)
                    metrics.end_time = time.monotonic()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.retries = task.retry_count

                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self._metrics.task_metrics[task_id] = metrics
                    self._emit("on_task_complete", task)
                    return result

                except Exception:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        metrics.end_time = time.monotonic()
                        metrics.duration = metrics.end_time - metrics.start_time
                        metrics.retries = task.retry_count - 1

                        task.status = TaskStatus.FAILED
                        self._metrics.task_metrics[task_id] = metrics
                        self._emit("on_task_fail", task)
                        return None

                    # Exponential backoff: 2^(retry_count-1) * 0.1 seconds.
                    backoff = (2 ** (task.retry_count - 1)) * 0.1
                    await asyncio.sleep(backoff)

        return None  # pragma: no cover

    # ------------------------------------------------------------------
    # Internal – event emission
    # ------------------------------------------------------------------

    def _emit(self, event: str, task: Task) -> None:
        """Invoke all registered listeners for *event*."""
        for listener in self._listeners.get(event, []):
            try:
                result = listener(task)
                # Support async listeners transparently.
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass  # Listener errors must not break the scheduler.
