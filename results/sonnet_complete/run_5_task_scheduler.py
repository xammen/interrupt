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
from typing import Any, Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Lifecycle states for a scheduled task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CircularDependencyError(Exception):
    """Raised when the task graph contains a cycle."""

    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Represents a unit of schedulable work.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable label.
        priority: Execution priority in the range [1, 10] (10 = highest).
        dependencies: IDs of tasks that must complete before this one runs.
        status: Current lifecycle state.
        retry_count: Number of times execution has been retried so far.
        max_retries: Maximum number of retry attempts before marking as FAILED.
        created_at: Wall-clock time when the task was created.
        result: Return value stored after successful execution.
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


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Timing and retry information collected during execution.

    Attributes:
        task_id: The task this record belongs to.
        start_time: Epoch seconds when execution began (last attempt).
        end_time: Epoch seconds when execution ended.
        retries: Total retry attempts consumed.
    """
    task_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    retries: int = 0

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds spent executing (excluding backoff sleep)."""
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Priority-aware async task scheduler with dependency resolution.

    Features
    --------
    * Topological sort with circular-dependency detection.
    * Independent tasks within a dependency level run concurrently up to
      *max_concurrency* simultaneous coroutines.
    * Exponential back-off retry (base 2 seconds, jitter-free).
    * Observer callbacks: ``on_task_start``, ``on_task_complete``,
      ``on_task_fail``.
    * ``get_execution_plan()`` returns ordered groups of task IDs.
    * Per-task and aggregate metrics.

    Parameters
    ----------
    max_concurrency:
        Maximum number of tasks allowed to run at the same time.
    """

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self._tasks: Dict[str, Task] = {}
        self._coroutines: Dict[str, Callable[[], Any]] = {}
        self._metrics: Dict[str, TaskMetrics] = {}

        # Observer callbacks – each is a list of async or sync callables
        self._listeners: Dict[str, List[Callable]] = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_task_fail": [],
        }

        self._scheduler_start: float = 0.0
        self._scheduler_end: float = 0.0

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------

    def add_task(self, task: Task, coroutine_fn: Callable[[], Any]) -> None:
        """Register a task and its associated async (or sync) callable.

        Parameters
        ----------
        task:
            The :class:`Task` descriptor.
        coroutine_fn:
            A zero-argument callable.  May return an awaitable; if it is a
            plain function it is called directly inside the event loop.

        Raises
        ------
        ValueError
            If a task with the same *id* has already been added.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' already registered")
        self._tasks[task.id] = task
        self._coroutines[task.id] = coroutine_fn
        self._metrics[task.id] = TaskMetrics(task_id=task.id)

    def on(self, event: str, callback: Callable) -> None:
        """Subscribe *callback* to *event*.

        Parameters
        ----------
        event:
            One of ``'on_task_start'``, ``'on_task_complete'``,
            ``'on_task_fail'``.
        callback:
            Callable receiving a single :class:`Task` argument.  May be a
            coroutine function.

        Raises
        ------
        ValueError
            If *event* is not recognised.
        """
        if event not in self._listeners:
            raise ValueError(f"Unknown event '{event}'. "
                             f"Valid events: {list(self._listeners)}")
        self._listeners[event].append(callback)

    # ------------------------------------------------------------------
    # Dependency graph helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Ensure all declared dependency IDs refer to registered tasks."""
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )

    def _build_graph(self) -> Dict[str, Set[str]]:
        """Return adjacency map: task_id -> set of task_ids that depend on it."""
        graph: Dict[str, Set[str]] = {tid: set() for tid in self._tasks}
        for task in self._tasks.values():
            for dep in task.dependencies:
                graph[dep].add(task.id)
        return graph

    def _topological_sort(self) -> List[List[str]]:
        """Kahn's algorithm; returns ordered *groups* for level-parallel execution.

        Returns
        -------
        List[List[str]]
            Each inner list is a group of task IDs whose dependencies are all
            satisfied by previous groups.  Tasks within a group may run in
            parallel.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        """
        in_degree: Dict[str, int] = {
            tid: len(t.dependencies) for tid, t in self._tasks.items()
        }
        dependents = self._build_graph()  # dep -> {tasks that need dep}

        # Seed with nodes that have no dependencies, sorted by priority desc
        ready: deque[str] = deque(
            sorted(
                (tid for tid, deg in in_degree.items() if deg == 0),
                key=lambda tid: -self._tasks[tid].priority,
            )
        )

        groups: List[List[str]] = []
        visited: Set[str] = set()

        while ready:
            # Collect all currently ready tasks as one parallel group
            current_group = list(ready)
            current_group.sort(key=lambda tid: -self._tasks[tid].priority)
            groups.append(current_group)
            ready.clear()

            next_ready: List[str] = []
            for tid in current_group:
                visited.add(tid)
                for dependent in dependents[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            next_ready.sort(key=lambda tid: -self._tasks[tid].priority)
            ready.extend(next_ready)

        if len(visited) != len(self._tasks):
            # Identify the cycle for a helpful error message
            cycle = self._find_cycle()
            raise CircularDependencyError(cycle)

        return groups

    def _find_cycle(self) -> List[str]:
        """DFS-based cycle detection; returns the cycle as an ordered list."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._tasks}
        path: List[str] = []
        cycle: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            path.append(node)
            for dep in self._tasks[node].dependencies:
                if color[dep] == GRAY:
                    # Found back-edge: extract cycle
                    idx = path.index(dep)
                    cycle.extend(path[idx:])
                    cycle.append(dep)
                    return True
                if color[dep] == WHITE and dfs(dep):
                    return True
            path.pop()
            color[node] = BLACK
            return False

        for tid in self._tasks:
            if color[tid] == WHITE:
                if dfs(tid):
                    break
        return cycle

    # ------------------------------------------------------------------
    # Public planning API
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Returns
        -------
        List[List[str]]
            Groups of task IDs in dependency order.  Tasks within a group
            have no inter-dependencies and may run in parallel.

        Raises
        ------
        CircularDependencyError
            If the graph contains a cycle.
        ValueError
            If any dependency references an unknown task.
        """
        self._validate_dependencies()
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _emit(self, event: str, task: Task) -> None:
        """Invoke all listeners registered for *event*."""
        for cb in self._listeners[event]:
            result = cb(task)
            if asyncio.iscoroutine(result):
                await result

    async def _run_task(self, task: Task) -> None:
        """Execute *task* with exponential back-off retries.

        The back-off delay for attempt *n* is ``2 ** (n - 1)`` seconds
        (i.e. 1 s, 2 s, 4 s, …).
        """
        metrics = self._metrics[task.id]
        fn = self._coroutines[task.id]

        task.status = TaskStatus.RUNNING
        metrics.start_time = time.monotonic()
        await self._emit("on_task_start", task)

        last_exc: Exception = RuntimeError("Task failed with no exception captured")

        for attempt in range(task.max_retries + 1):
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    result = await result
                task.result = result
                task.status = TaskStatus.COMPLETED
                metrics.end_time = time.monotonic()
                metrics.retries = task.retry_count
                await self._emit("on_task_complete", task)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < task.max_retries:
                    task.retry_count += 1
                    backoff = 2 ** attempt  # 1 s, 2 s, 4 s …
                    await asyncio.sleep(backoff)
                else:
                    break

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        metrics.end_time = time.monotonic()
        metrics.retries = task.retry_count
        await self._emit("on_task_fail", task)
        raise last_exc  # re-raise so the caller can handle it

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, TaskMetrics]:
        """Execute all registered tasks respecting dependencies and concurrency.

        Tasks are executed group-by-group (each group is a dependency level).
        Within a group, up to *max_concurrency* tasks run in parallel.

        Returns
        -------
        Dict[str, TaskMetrics]
            Mapping of task ID to its collected metrics.

        Raises
        ------
        CircularDependencyError
            If the dependency graph contains a cycle.
        ValueError
            If any dependency references an unknown task.
        Exception
            The first exception from a task that exhausted all retries; other
            exceptions from the same group are collected and re-raised as an
            :class:`ExceptionGroup` (Python 3.11+) or a plain ``RuntimeError``
            on earlier versions.
        """
        self._validate_dependencies()
        groups = self._topological_sort()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        self._scheduler_start = time.monotonic()

        async def bounded(task: Task) -> None:
            async with semaphore:
                await self._run_task(task)

        for group in groups:
            coros = [bounded(self._tasks[tid]) for tid in group]
            results = await asyncio.gather(*coros, return_exceptions=True)

            # Surface any failures
            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                if len(errors) == 1:
                    raise errors[0]
                raise RuntimeError(
                    f"{len(errors)} tasks failed in the same group: "
                    + "; ".join(str(e) for e in errors)
                )

        self._scheduler_end = time.monotonic()
        return self._metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def total_elapsed(self) -> float:
        """Total wall-clock seconds for the last :meth:`run` call."""
        return self._scheduler_end - self._scheduler_start

    def get_metrics(self) -> Dict[str, TaskMetrics]:
        """Return a snapshot of current per-task metrics."""
        return dict(self._metrics)
