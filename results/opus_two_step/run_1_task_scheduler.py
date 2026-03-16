"""
Async task scheduler with priority queue, dependency resolution,
concurrent execution, retry logic, and observer-pattern event emission.
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
    """Possible states of a scheduled task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the task graph."""


@dataclass
class Task:
    """Represents a unit of work to be scheduled and executed.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable task name.
        priority: Execution priority from 1 (lowest) to 10 (highest).
        dependencies: List of task IDs that must complete before this task runs.
        status: Current lifecycle status of the task.
        retry_count: Number of times this task has been retried after failure.
        max_retries: Maximum number of retry attempts before permanent failure.
        created_at: Timestamp of task creation.
        result: The return value of the task coroutine, if completed.
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


# Type alias for the async callable that a task will execute.
TaskCoroutine = Callable[[Task], Coroutine[Any, Any, Any]]

# Type alias for event listener callbacks.
EventListener = Callable[[Task], Any]


class TaskScheduler:
    """Async task scheduler with dependency resolution, concurrency control,
    retry logic with exponential backoff, and an observer-pattern event system.

    Args:
        max_concurrency: Maximum number of tasks that may run in parallel.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._max_concurrency: int = max_concurrency
        self._tasks: dict[str, Task] = {}
        self._coroutines: dict[str, TaskCoroutine] = {}
        self._listeners: dict[str, list[EventListener]] = defaultdict(list)
        self._metrics: TaskMetrics = TaskMetrics()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coro: TaskCoroutine) -> None:
        """Register a task and its associated coroutine with the scheduler.

        Args:
            task: The Task instance to schedule.
            coro: An async callable that receives the Task and performs the work.

        Raises:
            ValueError: If a task with the same ID is already registered.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coro

    # ------------------------------------------------------------------
    # Observer / event system
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

    async def _emit(self, event: str, task: Task) -> None:
        """Emit an event to all registered listeners.

        If a listener is a coroutine function its result is awaited; plain
        callables are invoked synchronously.
        """
        for listener in self._listeners.get(event, []):
            try:
                ret = listener(task)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception:
                # Listener errors must not crash the scheduler.
                pass

    # ------------------------------------------------------------------
    # Dependency resolution (topological sort)
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure every listed dependency references a registered task.

        Raises:
            ValueError: If a dependency ID does not correspond to a known task.
        """
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _topological_sort(self) -> list[list[str]]:
        """Compute a layered topological ordering of all registered tasks.

        Tasks in the same layer have no mutual dependencies and can run
        concurrently.  Within a layer tasks are sorted by descending priority.

        Returns:
            A list of groups (layers), each group being a list of task IDs.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
        """
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: dict[str, list[str]] = defaultdict(list)

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task.id)
                in_degree[task.id] += 1

        # Seed the queue with tasks that have no unmet dependencies.
        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )

        layers: list[list[str]] = []
        visited_count = 0

        while queue:
            # All items currently in the queue form one concurrent layer.
            layer_size = len(queue)
            layer: list[str] = []
            for _ in range(layer_size):
                tid = queue.popleft()
                layer.append(tid)
                visited_count += 1
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        queue.append(dep_tid)

            # Sort layer by descending priority so higher-priority tasks start first.
            layer.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            layers.append(layer)

        if visited_count != len(self._tasks):
            # Identify the tasks involved in the cycle for a helpful message.
            remaining = {tid for tid, deg in in_degree.items() if deg > 0}
            raise CircularDependencyError(
                f"Circular dependency detected among tasks: {remaining}"
            )

        return layers

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> list[list[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that may execute concurrently.
        Groups are returned in the order they must be executed.

        Returns:
            A list of execution groups (lists of task IDs).

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            ValueError: If any dependency references an unknown task.
        """
        self._validate_dependencies()
        return self._topological_sort()

    @property
    def metrics(self) -> TaskMetrics:
        """Access the collected execution metrics."""
        return self._metrics

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    async def _run_task(self, task: Task, semaphore: asyncio.Semaphore) -> None:
        """Execute a single task with retry and exponential backoff.

        Args:
            task: The task to execute.
            semaphore: Semaphore enforcing the concurrency limit.
        """
        async with semaphore:
            while True:
                task.status = TaskStatus.RUNNING
                await self._emit("on_task_start", task)

                start = time.monotonic()
                try:
                    result = await self._coroutines[task.id](task)
                    elapsed = time.monotonic() - start

                    task.status = TaskStatus.COMPLETED
                    task.result = result
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

                    # Exponential backoff: 2^(retry - 1) seconds, capped at 60 s.
                    backoff = min(2 ** (task.retry_count - 1), 60)
                    await asyncio.sleep(backoff)

    async def run(self) -> TaskMetrics:
        """Execute all registered tasks respecting dependencies and concurrency.

        Returns:
            A ``TaskMetrics`` instance with timing and retry information.

        Raises:
            CircularDependencyError: If the dependency graph contains a cycle.
            ValueError: If any dependency references an unknown task.
        """
        self._validate_dependencies()
        layers = self._topological_sort()

        semaphore = asyncio.Semaphore(self._max_concurrency)
        self._metrics = TaskMetrics()

        overall_start = time.monotonic()

        for layer in layers:
            # Check that all dependencies in this layer completed successfully.
            tasks_to_run: list[Task] = []
            for tid in layer:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_ok:
                    tasks_to_run.append(task)
                else:
                    # A dependency failed; mark this task as failed immediately.
                    task.status = TaskStatus.FAILED
                    self._metrics.per_task_time[task.id] = 0.0
                    self._metrics.retry_counts[task.id] = 0
                    await self._emit("on_task_fail", task)

            # Run all eligible tasks in this layer concurrently.
            if tasks_to_run:
                await asyncio.gather(
                    *(self._run_task(t, semaphore) for t in tasks_to_run)
                )

        self._metrics.total_time = time.monotonic() - overall_start
        return self._metrics
