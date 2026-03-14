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
        # in-degree: number of unresolved dependencies per task
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        graph = self._build_graph()

        for task in self._tasks.values():
            for dep in task.dependencies:
                in_degree[task.id] += 1

        # Seed the queue with tasks that have no dependencies,
        # sorted by descending priority so high-priority tasks execute first.
        queue: deque[str] = deque(
            sorted(
                (tid for tid, deg in in_degree.items() if deg == 0),
                key=lambda tid: self._tasks[tid].priority,
                reverse=True,
            )
        )

        groups: List[List[str]] = []
        visited: Set[str] = set()

        while queue:
            # Collect all tasks that are currently unblocked as one group
            group_size = len(queue)
            group: List[str] = []
            for _ in range(group_size):
                tid = queue.popleft()
                group.append(tid)
                visited.add(tid)

            # Sort the group by descending priority
            group.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            groups.append(group)

            # Unblock dependents
            next_ready: List[str] = []
            for tid in group:
                for dependent in graph[tid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            # Add newly unblocked tasks sorted by priority
            next_ready.sort(key=lambda tid: self._tasks[tid].priority, reverse=True)
            queue.extend(next_ready)

        if len(visited) != len(self._tasks):
            # Find a cycle via DFS for a helpful error message
            cycle = self._find_cycle()
            raise CircularDependencyError(cycle)

        return groups

    def _find_cycle(self) -> List[str]:
        """Return a list of task IDs that form a cycle (for error reporting)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in self._tasks}
        parent: Dict[str, Optional[str]] = {tid: None for tid in self._tasks}
        cycle: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for task in self._tasks.values():
                if node in task.dependencies:
                    # node -> task.id edge (node must complete before task.id)
                    pass
            # Walk forward edges: node's dependents
            for task in self._tasks.values():
                if node in task.dependencies:
                    child = task.id
                    if color[child] == GRAY:
                        # Reconstruct cycle
                        cycle.append(child)
                        cur: Optional[str] = node
                        while cur is not None and cur != child:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(child)
                        cycle.reverse()
                        return True
                    if color[child] == WHITE:
                        parent[child] = node
                        if dfs(child):
                            return True
            color[node] = BLACK
            return False

        for tid in self._tasks:
            if color[tid] == WHITE:
                if dfs(tid):
                    break

        return cycle if cycle else list(self._tasks.keys())[:2]

    # ------------------------------------------------------------------
    # Observer helpers
    # ------------------------------------------------------------------

    async def _emit(self, event: str, task: Task) -> None:
        """Invoke all registered listeners for *event* with *task*."""
        for cb in self._listeners[event]:
            result = cb(task)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute_task(self, task: Task) -> None:
        """Run a single task with retry/backoff logic.

        The task's status is updated throughout execution.  On each failed
        attempt the scheduler sleeps for ``2 ** attempt`` seconds before
        retrying.  After exhausting *max_retries* the task is marked FAILED.
        """
        metrics = self._metrics[task.id]
        fn = self._coroutines[task.id]

        task.status = TaskStatus.RUNNING
        metrics.start_time = time.monotonic()
        await self._emit("on_task_start", task)

        last_exc: Optional[Exception] = None

        for attempt in range(task.max_retries + 1):
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    task.result = await result
                else:
                    task.result = result

                task.status = TaskStatus.COMPLETED
                metrics.end_time = time.monotonic()
                metrics.retries = attempt
                await self._emit("on_task_complete", task)
                return

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                task.retry_count += 1
                metrics.retries = task.retry_count

                if attempt < task.max_retries:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        metrics.end_time = time.monotonic()
        await self._emit("on_task_fail", task)
        raise last_exc  # type: ignore[misc]

    async def _run_group(
        self, group: List[str], semaphore: asyncio.Semaphore
    ) -> List[str]:
        """Execute all tasks in *group* concurrently, bounded by *semaphore*.

        Returns the IDs of tasks that failed so callers can halt if needed.
        """
        failed: List[str] = []

        async def _guarded(tid: str) -> None:
            async with semaphore:
                try:
                    await self._execute_task(self._tasks[tid])
                except Exception:  # noqa: BLE001
                    failed.append(tid)

        await asyncio.gather(*(_guarded(tid) for tid in group))
        return failed

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute all registered tasks respecting dependencies and priority.

        Tasks whose dependencies have all completed are grouped and run
        concurrently (up to *max_concurrency*).  If any task in a group fails
        after exhausting its retries, the entire remaining schedule is aborted
        and the method returns immediately with partial results.

        Returns
        -------
        Dict[str, Any]
            Mapping of task ID to its ``result`` value (``None`` for failed or
            un-executed tasks).

        Raises
        ------
        CircularDependencyError
            If the dependency graph is cyclic.
        ValueError
            If any dependency ID is unknown.
        """
        self._validate_dependencies()
        groups = self._topological_sort()

        semaphore = asyncio.Semaphore(self.max_concurrency)
        self._scheduler_start = time.monotonic()

        for group in groups:
            failed = await self._run_group(group, semaphore)
            if failed:
                # Abort – don't execute dependent groups
                break

        self._scheduler_end = time.monotonic()
        return {tid: task.result for tid, task in self._tasks.items()}

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_execution_plan(self) -> List[List[str]]:
        """Return the ordered execution groups without running anything.

        Each inner list contains task IDs that can run in parallel.  Groups
        are ordered by dependency level; within a group tasks are sorted by
        descending priority.

        Raises
        ------
        CircularDependencyError
            If the dependency graph is cyclic.
        ValueError
            If any dependency ID is unknown.
        """
        self._validate_dependencies()
        return self._topological_sort()

    def get_metrics(self) -> Dict[str, TaskMetrics]:
        """Return per-task :class:`TaskMetrics` records (keyed by task ID)."""
        return dict(self._metrics)

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Return the current :class:`TaskStatus` for *task_id*.

        Raises
        ------
        KeyError
            If *task_id* is not registered.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task id: '{task_id}'")
        return self._tasks[task_id].status

    @property
    def total_elapsed(self) -> float:
        """Total wall-clock seconds from the start of :meth:`run` to its end."""
        return self._scheduler_end - self._scheduler_start

    def summary(self) -> Dict[str, Any]:
        """Return a high-level execution summary dict.

        Keys
        ----
        total_tasks, completed, failed, pending, running, total_elapsed_seconds
        """
        statuses = [t.status for t in self._tasks.values()]
        return {
            "total_tasks": len(self._tasks),
            "completed": statuses.count(TaskStatus.COMPLETED),
            "failed": statuses.count(TaskStatus.FAILED),
            "pending": statuses.count(TaskStatus.PENDING),
            "running": statuses.count(TaskStatus.RUNNING),
            "total_elapsed_seconds": round(self.total_elapsed, 4),
        }
