# The Prompt (identical for every run)

Write a Python module called `task_scheduler.py` that implements an async task scheduler with the following features:

1. A `Task` dataclass with fields: `id` (str), `name` (str), `priority` (int 1-10), `dependencies` (list of task IDs), `status` (enum: PENDING, RUNNING, COMPLETED, FAILED), `retry_count` (int), `max_retries` (int, default 3), `created_at` (datetime), `result` (Optional[Any])

2. A `TaskScheduler` class that:
   - Maintains a priority queue of tasks
   - Resolves dependency graphs before execution (topological sort)
   - Detects circular dependencies and raises `CircularDependencyError`
   - Runs independent tasks concurrently using `asyncio`
   - Implements exponential backoff retry logic for failed tasks
   - Has a configurable concurrency limit (max parallel tasks)
   - Emits events via an observer pattern (on_task_start, on_task_complete, on_task_fail)
   - Provides a `get_execution_plan()` method that returns the ordered execution groups
   - Tracks execution metrics (total time, per-task time, retry count)

3. Include proper error handling, type hints, and docstrings.

4. Write at least 5 unit tests using pytest that cover:
   - Basic task execution
   - Dependency resolution
   - Circular dependency detection
   - Retry logic with exponential backoff
   - Concurrent execution respecting concurrency limits

Write the complete code in a single response. Do not ask questions, just write the code.
