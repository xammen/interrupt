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
        self._validate_dependencies()

        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        queue: deque[str] = deque()
        for tid, degree in in_degree.items():
            if degree == 0:
                queue.append(tid)

        groups: list[list[str]] = []
        processed = 0

        while queue:
            # All tasks currently in the queue form one concurrent group.
            # Sort within the group by priority (highest first) for determinism.
            group = sorted(queue, key=lambda tid: self._tasks[tid].priority, reverse=True)
            queue.clear()
            groups.append(group)
            processed += len(group)

            for tid in group:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        queue.append(dep_tid)

        if processed != len(self._tasks):
            # Find a cycle for a helpful error message.
            cycle = self._find_cycle()
            raise CircularDependencyError(cycle)

        return groups

    def _find_cycle(self) -> list[str]:
        """Return one cycle from the dependency graph (DFS-based)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for dep_id in self._tasks[node].dependencies:
                if color[dep_id] == GRAY:
                    # Reconstruct the cycle.
                    cycle = [dep_id, node]
                    cur: str = node
                    p = parent.get(cur)
                    while p is not None and p != dep_id:
                        cur = p
                        cycle.append(cur)
                        p = parent.get(cur)
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
                cycle = dfs(tid)
                if cycle:
                    return cycle
        return []

    # ── Execution ────────────────────────────────────────────────────

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can execute concurrently.
        The outer list is ordered: group *i* must finish before group *i+1* starts.

        Returns:
            Ordered list of concurrent execution groups.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        return self._topological_sort()

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry + exponential backoff.

        Args:
            task: The task to execute.
            semaphore: Controls concurrency.
        """
        coro_fn = self._coroutines[task.id]

        while True:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)

                start = time.monotonic()
                try:
                    task.result = await coro_fn(task)
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
                    # Exponential backoff: 2^(retry_count - 1) seconds
                    # e.g. 1s, 2s, 4s, ...
                    backoff = 2 ** (task.retry_count - 1)
                    await asyncio.sleep(backoff)

    async def run(self) -> TaskMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A :class:`TaskMetrics` instance with timing and retry information.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        self._metrics = TaskMetrics()

        overall_start = time.monotonic()

        for group in groups:
            tasks_in_group = [self._tasks[tid] for tid in group]

            # Skip tasks whose dependencies failed.
            runnable: list[Task] = []
            for task in tasks_in_group:
                dep_failed = any(
                    self._tasks[d].status == TaskStatus.FAILED
                    for d in task.dependencies
                )
                if dep_failed:
                    task.status = TaskStatus.FAILED
                    self._metrics.retry_counts[task.id] = 0
                    await self._emit("on_task_fail", task)
                else:
                    runnable.append(task)

            await asyncio.gather(
                *(self._run_task(t, semaphore) for t in runnable)
            )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics
