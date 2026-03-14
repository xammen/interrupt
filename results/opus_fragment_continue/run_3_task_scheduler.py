"""Async task scheduler with dependency resolution, retry logic, and observer pattern."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class TaskStatus(Enum):
    """Possible states for a scheduled task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected among tasks."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        self.cycle = cycle or []
        msg = "Circular dependency detected"
        if self.cycle:
            msg += f": {' -> '.join(self.cycle)}"
        super().__init__(msg)


@dataclass
class Task:
    """Represents a unit of work to be scheduled.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        priority: Execution priority (1 = lowest, 10 = highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current execution status.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum retry attempts before marking as FAILED.
        created_at: Timestamp when the task was created.
        result: The return value of the task coroutine, or None.
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


class TaskScheduler:
    """Async task scheduler with dependency resolution and concurrency control.

    Args:
        max_concurrency: Maximum number of tasks that can run in parallel.

    Example::

        scheduler = TaskScheduler(max_concurrency=4)
        scheduler.add_task(Task(id="a", name="Step A"), coro=my_coroutine)
        scheduler.add_task(Task(id="b", name="Step B", dependencies=["a"]), coro=other_coro)
        metrics = await scheduler.run()
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._metrics = TaskMetrics()

    # ── Task management ──────────────────────────────────────────────

    def add_task(
        self,
        task: Task,
        coro: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a task and its associated coroutine function.

        Args:
            task: The Task dataclass instance.
            coro: An async callable that performs the actual work.
                  It receives the Task as its sole argument.

        Raises:
            ValueError: If a task with the same ID already exists.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    def get_task(self, task_id: str) -> Task:
        """Return a task by its ID.

        Raises:
            KeyError: If the task ID is not found.
        """
        return self._tasks[task_id]

    # ── Observer pattern ─────────────────────────────────────────────

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register an event listener.

        Supported events: ``on_task_start``, ``on_task_complete``, ``on_task_fail``.

        Args:
            event: Event name.
            callback: A callable (sync or async) invoked with the relevant Task.
        """
        self._listeners[event].append(callback)

    async def _emit(self, event: str, task: Task) -> None:
        """Emit an event to all registered listeners."""
        for callback in self._listeners.get(event, []):
            ret = callback(task)
            if asyncio.iscoroutine(ret):
                await ret

    # ── Dependency resolution ────────────────────────────────────────

    def _validate_dependencies(self) -> None:
        """Ensure all dependency references point to known tasks.

        Raises:
            KeyError: If a dependency references a task ID that was not registered.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise KeyError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _topological_sort(self) -> list[list[str]]:
        """Return execution groups via Kahn's algorithm (topological sort).

        Tasks within the same group have no interdependencies and can run
        concurrently.  Groups themselves must be executed in order.

        Returns:
            A list of groups, where each group is a list of task IDs.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                in_degree[task.id] += 1
                dependents[dep_id].append(task.id)

        queue: deque[str] = deque()
        for tid, degree in in_degree.items():
            if degree == 0:
                queue.append(tid)

        groups: list[list[str]] = []
        visited_count = 0

        while queue:
            # All tasks currently in the queue belong to the same execution group
            group = sorted(queue, key=lambda t: self._tasks[t].priority, reverse=True)
            queue.clear()
            groups.append(group)
            visited_count += len(group)

            for tid in group:
                for dependent_id in dependents[tid]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)

        if visited_count != len(self._tasks):
            # Detect the actual cycle for a better error message
            cycle = self._find_cycle()
            raise CircularDependencyError(cycle)

        return groups

    def _find_cycle(self) -> list[str]:
        """Find and return one cycle in the dependency graph using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for dep_id in self._tasks[node].dependencies:
                if color[dep_id] == GRAY:
                    # Back edge found – reconstruct cycle
                    cycle = [dep_id, node]
                    cur = parent[node]
                    while cur is not None and cur != dep_id:
                        cycle.append(cur)
                        cur = parent[cur]
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
                result = dfs(tid)
                if result is not None:
                    return result
        return []

    # ── Execution ────────────────────────────────────────────────────

    async def _run_task(self, task_id: str, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry logic, respecting concurrency limits.

        Args:
            task_id: The ID of the task to run.
            semaphore: Semaphore controlling maximum concurrency.
        """
        task = self._tasks[task_id]
        coro_fn = self._coroutines[task_id]

        # Skip tasks already marked FAILED by _cancel_dependents
        if task.status == TaskStatus.FAILED:
            return

        async with semaphore:
            while True:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)
                start = time.monotonic()

                try:
                    task.result = await coro_fn(task)
                    elapsed = time.monotonic() - start
                    task.status = TaskStatus.COMPLETED
                    self._metrics.per_task_time[task_id] = elapsed
                    self._metrics.retry_counts[task_id] = task.retry_count
                    await self._emit("on_task_complete", task)
                    return
                except Exception:
                    elapsed = time.monotonic() - start
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        task.status = TaskStatus.FAILED
                        self._metrics.per_task_time[task_id] = elapsed
                        self._metrics.retry_counts[task_id] = task.retry_count
                        await self._emit("on_task_fail", task)
                        return
                    # Brief back-off before retry
                    await asyncio.sleep(0.05 * task.retry_count)

    async def run(self) -> TaskMetrics:
        """Validate, sort, and execute all registered tasks.

        Returns:
            TaskMetrics with timing and retry information.

        Raises:
            KeyError: If any dependency references an unknown task.
            CircularDependencyError: If a dependency cycle exists.
        """
        self._validate_dependencies()
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        overall_start = time.monotonic()

        for group in groups:
            aws = [self._run_task(tid, semaphore) for tid in group]
            await asyncio.gather(*aws)

            # If any task in this group failed, skip downstream dependents
            failed_ids = {
                tid for tid in group if self._tasks[tid].status == TaskStatus.FAILED
            }
            if failed_ids:
                self._cancel_dependents(failed_ids)

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics

    def _cancel_dependents(self, failed_ids: set[str]) -> None:
        """Mark all transitive dependents of *failed_ids* as FAILED.

        This prevents downstream tasks from executing when an upstream
        dependency has failed.
        """
        queue: deque[str] = deque(failed_ids)
        visited: set[str] = set(failed_ids)

        while queue:
            current = queue.popleft()
            for task in self._tasks.values():
                if task.id not in visited and current in task.dependencies:
                    task.status = TaskStatus.FAILED
                    visited.add(task.id)
                    queue.append(task.id)

    # ── Introspection helpers ────────────────────────────────────────

    @property
    def tasks(self) -> list[Task]:
        """Return a snapshot of all registered tasks sorted by priority (desc)."""
        return sorted(self._tasks.values(), key=lambda t: t.priority, reverse=True)

    @property
    def metrics(self) -> TaskMetrics:
        """Return the current metrics object."""
        return self._metrics

    def reset(self) -> None:
        """Reset all tasks to PENDING and clear metrics."""
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.retry_count = 0
            task.result = None
        self._metrics = TaskMetrics()
