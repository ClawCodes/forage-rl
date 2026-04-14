"""Shared helpers for process-based experiment concurrency."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import TypeVar

from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    is_neural_agent as _is_registered_neural_agent,
)
from forage_rl.utils.torch_support import resolve_device


TAgentLike = TypeVar("TAgentLike", Agent, EvaluatorSpec)


@dataclass(frozen=True)
class ExecutionStrategy:
    worker_count: int
    device: str
    uses_torch: bool
    mp_context: mp.context.BaseContext | None
    worker_note: str | None = None


def is_neural_agent(agent: Agent) -> bool:
    """Return whether an agent is backed by PyTorch."""
    return _is_registered_neural_agent(agent)


def uses_torch_agents(items: list[Agent | EvaluatorSpec] | None) -> bool:
    """Return whether a list of agents or evaluator specs includes a neural agent."""
    if items is None:
        return False

    for item in items:
        agent = item if isinstance(item, Agent) else item.agent
        if is_neural_agent(agent):
            return True
    return False


def split_torch_items(
    items: list[TAgentLike] | None,
) -> tuple[list[TAgentLike], list[TAgentLike]]:
    """Split items into CPU-safe and torch-backed groups, preserving order."""
    if items is None:
        return [], []

    cpu_items: list[TAgentLike] = []
    torch_items: list[TAgentLike] = []
    for item in items:
        agent = item if isinstance(item, Agent) else item.agent
        if is_neural_agent(agent):
            torch_items.append(item)
        else:
            cpu_items.append(item)
    return cpu_items, torch_items


def resolve_worker_count(task_count: int, workers: int | None = None) -> int:
    """Resolve the number of workers to use for a task batch."""
    if task_count < 0:
        raise ValueError(f"task_count must be >= 0, got {task_count}")

    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    if task_count == 0:
        return 0

    if workers is None:
        return min(task_count, os.cpu_count() or 1)

    return min(task_count, workers)


def resolve_execution_strategy(
    task_count: int,
    workers: int | None = None,
    *,
    uses_torch: bool = False,
    device: str = "auto",
) -> ExecutionStrategy:
    """Resolve device, worker count, and process context for a task batch."""
    resolved_device = resolve_device(device)
    worker_count = resolve_worker_count(task_count, workers)
    mp_context = None
    worker_note = None

    if uses_torch:
        mp_context = mp.get_context("spawn")
        if resolved_device in {"cuda", "mps"} and worker_count > 1:
            worker_count = 1
            worker_note = (
                f"Clamped neural workload to one worker on {resolved_device}."
            )

    return ExecutionStrategy(
        worker_count=worker_count,
        device=resolved_device,
        uses_torch=uses_torch,
        mp_context=mp_context,
        worker_note=worker_note,
    )
