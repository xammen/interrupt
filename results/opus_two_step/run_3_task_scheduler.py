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
    """Status of a task in the scheduler."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        if cycle:
            msg = f"Circular dependency detected: {' -> '.join(cycle)}"
        else:
            msg = "Circular dependency detected in task graph"
        super().__init__(msg)
        self.cycle = cycle


@dataclass
class Task:
    """Represents a schedulable unit of work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority from 1 (lowest) to 10 (highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum allowed retry attempts.
        created_at: Timestamp when the task was created.
        result: The return value of the task's coroutine, if completed.
    """

    id: str
    name: str
    priority: int
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
        retry_counts: Mapping of task ID to the number of retries performed.
    """

    total_time: float = 0.0
    per_task_time: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)


# Type alias for the async callable that a task executes.
TaskCoroutine = Callable[[Task], Coroutine[Any, Any, Any]]

# Type alias for event listener callbacks.
EventListener = Callable[[Task], Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control, and retry logic.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
        base_backoff: Base delay in seconds for exponential backoff retries.
    """

    def __init__(self, max_concurrency: int = 4, base_backoff: float = 1.0) -> None:
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, TaskCoroutine] = {}
        self._max_concurrency = max_concurrency
        self._base_backoff = base_backoff
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self._metrics = TaskMetrics()

    # -- Task registration ---------------------------------------------------

    def add_task(self, task: Task, coro: TaskCoroutine) -> None:
        """Register a task and its associated coroutine with the scheduler.

        Args:
            task: The Task instance to schedule.
            coro: An async callable that receives the Task and performs work.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the scheduler.

        Args:
            task_id: The ID of the task to remove.

        Raises:
            KeyError: If the task ID is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        del self._tasks[task_id]
        del self._coroutines[task_id]

    # -- Observer pattern -----------------------------------------------------

    def on(self, event: str, listener: EventListener) -> None:
        """Subscribe a listener to a scheduler event.

        Supported events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.

        Args:
            event: The event name.
            listener: A callable (sync or async) invoked with the relevant Task.
        """
        self._listeners[event].append(listener)

    async def _emit(self, event: str, task: Task) -> None:
        """Emit an event, calling all registered listeners."""
        for listener in self._listeners.get(event, []):
            ret = listener(task)
            if asyncio.iscoroutine(ret) or asyncio.isfuture(ret):
                await ret

    # -- Dependency graph utilities -------------------------------------------

    def _build_graph(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Build adjacency list and in-degree map from registered tasks.

        Returns:
            A tuple of (adjacency list, in-degree mapping).
        """
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    raise KeyError(
                        f"Task '{tid}' depends on unknown task '{dep}'"
                    )
                adj[dep].append(tid)
                in_degree[tid] += 1

        return adj, in_degree

    def _topological_sort(self) -> list[list[str]]:
        """Perform a topological sort, grouping tasks into execution layers.

        Each layer contains tasks whose dependencies are satisfied by all
        preceding layers. Within a layer, tasks are sorted by descending
        priority so higher-priority work is attempted first.

        Returns:
            Ordered list of groups (layers), each group is a list of task IDs.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        adj, in_degree = self._build_graph()

        queue: deque[str] = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)

        layers: list[list[str]] = []
        visited_count = 0

        while queue:
            layer = sorted(queue, key=lambda t: self._tasks[t].priority, reverse=True)
            layers.append(layer)
            visited_count += len(layer)

            next_queue: deque[str] = deque()
            for tid in layer:
                for neighbour in adj[tid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        next_queue.append(neighbour)
            queue = next_queue

        if visited_count != len(self._tasks):
            # Find the cycle for a more helpful error message.
            cycle = self._find_cycle(adj)
            raise CircularDependencyError(cycle)

        return layers

    @staticmethod
    def _find_cycle(adj: dict[str, list[str]]) -> list[str] | None:
        """Detect and return one cycle in the directed graph, or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = defaultdict(int)
        parent: dict[str, str] = {}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for nbr in adj.get(node, []):
                if color[nbr] == GRAY:
                    # Reconstruct cycle.
                    cycle = [nbr, node]
                    cur = node
                    while cur != nbr:
                        cur = parent.get(cur, nbr)
                        if cur == nbr:
                            break
                        cycle.append(cur)
                    cycle.reverse()
                    cycle.append(nbr)
                    return cycle
                if color[nbr] == WHITE:
                    parent[nbr] = node
                    result = dfs(nbr)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for node in list(adj.keys()):
            if color[node] == WHITE:
                result = dfs(node)
                if result:
                    return result
        return None

    # -- Execution plan -------------------------------------------------------

    def get_execution_plan(self) -> list[list[Task]]:
        """Return the ordered execution groups without running anything.

        Each inner list is a group of tasks that can run concurrently. Groups
        are ordered so that all dependencies of a group are in earlier groups.

        Returns:
            List of task groups.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        layers = self._topological_sort()
        return [[self._tasks[tid] for tid in layer] for layer in layers]

    # -- Execution ------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry and backoff logic.

        Args:
            task: The task to run.
            semaphore: Controls concurrency.
        """
        coro = self._coroutines[task.id]

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)

                start = time.monotonic()
                try:
                    task.result = await coro(task)
                    elapsed = time.monotonic() - start

                    task.status = TaskStatus.COMPLETED
                    self._metrics.per_task_time[task.id] = elapsed
                    self._metrics.retry_counts[task.id] = task.retry_count
                    await self._emit("on_task_complete", task)
                    return

                except Exception:
                    elapsed = time.monotonic() - start
                    task.retry_count += 1

                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        self._metrics.per_task_time[task.id] = elapsed
                        self._metrics.retry_counts[task.id] = task.retry_count
                        await self._emit("on_task_fail", task)
                        return

                    # Exponential backoff before retry.
                    delay = self._base_backoff * (2 ** (task.retry_count - 1))
                    task.status = TaskStatus.PENDING
                    await asyncio.sleep(delay)

    async def run(self) -> TaskMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A TaskMetrics instance summarising the execution.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        self._metrics = TaskMetrics()
        layers = self._topological_sort()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        overall_start = time.monotonic()

        for layer in layers:
            tasks_in_layer = [self._tasks[tid] for tid in layer]
            await asyncio.gather(
                *(self._run_task(t, semaphore) for t in tasks_in_layer)
            )

            # If any dependency in this layer failed, mark dependants as failed.
            failed_ids = {t.id for t in tasks_in_layer if t.status == TaskStatus.FAILED}
            if failed_ids:
                self._propagate_failures(failed_ids)

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    def _propagate_failures(self, failed_ids: set[str]) -> None:
        """Mark all transitive dependants of *failed_ids* as FAILED."""
        adj, _ = self._build_graph()
        visited: set[str] = set()
        queue = deque(failed_ids)

        while queue:
            current = queue.popleft()
            for dep in adj.get(current, []):
                if dep not in visited and self._tasks[dep].status == TaskStatus.PENDING:
                    self._tasks[dep].status = TaskStatus.FAILED
                    self._metrics.per_task_time.setdefault(dep, 0.0)
                    self._metrics.retry_counts.setdefault(dep, 0)
                    visited.add(dep)
                    queue.append(dep)

    # -- Introspection --------------------------------------------------------

    @property
    def metrics(self) -> TaskMetrics:
        """Return the metrics from the most recent run."""
        return self._metrics

    @property
    def tasks(self) -> dict[str, Task]:
        """Return the internal task registry (read-only view)."""
        return dict(self._tasks)
