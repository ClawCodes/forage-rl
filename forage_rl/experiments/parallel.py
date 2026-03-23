"""Shared helpers for process-based experiment concurrency."""

import os


def resolve_worker_count(task_count: int, workers: int | None = None) -> int:
    """Resolve the number of workers to use for a task batch."""
    if task_count < 0:
        raise ValueError(f"task_count must be >= 0, got {task_count}")

    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    if task_count == 0:
        return 0

    if workers is None:
        return min(task_count, os.cpu_count() or 1, 8)

    return min(task_count, workers)
