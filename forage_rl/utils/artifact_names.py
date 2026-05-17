"""Pure artifact naming helpers shared by file I/O code."""

from __future__ import annotations

import re
from pathlib import Path

from forage_rl.agents.context import (
    DEFAULT_NEURAL_CONTEXT_MODE,
    context_mode_artifact_label,
)
from forage_rl.agents.registry import (
    Agent,
    is_neural_agent,
)
from forage_rl.agents.identities import (
    EvaluatorMode,
    EvaluatorIdentity,
    resolve_agent_context_mode,
)
from forage_rl.environments import builtin_maze_horizon, resolve_effective_horizon


def obs_tag(observable: bool) -> str:
    return "FO" if observable else "PO"


def horizon_suffix(maze_name: str, horizon: int | None = None) -> str:
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    if resolved_horizon == builtin_maze_horizon(maze_name):
        return ""
    return f"_h{resolved_horizon}"


def artifact_prefix(
    maze_name: str,
    observable: bool,
    horizon: int | None = None,
) -> str:
    return f"{maze_name}_{obs_tag(observable)}{horizon_suffix(maze_name, horizon)}"


def neural_context_suffix(agent: Agent, context_mode: str | None = None) -> str:
    effective_context = resolve_agent_context_mode(agent, context_mode)
    if (
        effective_context is None
        or effective_context == DEFAULT_NEURAL_CONTEXT_MODE
    ):
        return ""
    return f"_{context_mode_artifact_label(effective_context)}"


def extract_run_id(filepath: Path) -> int:
    match = re.search(r"_(\d+)\.npz$", filepath.name)
    if match is None:
        raise ValueError(f"Could not parse run id from filename: {filepath.name}")
    return int(match.group(1))


def is_canonical_run_dataset_file(path: Path) -> bool:
    return re.search(r"_run_dataset_\d+\.npz$", path.name) is not None


def matches_exact_horizon_prefix(
    path: Path,
    prefix: str,
    *,
    horizon: int | None,
) -> bool:
    if not path.name.startswith(f"{prefix}_"):
        return False
    if horizon is not None:
        return True
    remainder = path.name[len(prefix) + 1 :]
    return re.match(r"h\d+_", remainder) is None


def run_dataset_filename(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool,
    context_mode: str | None = None,
    horizon: int | None = None,
) -> str:
    return (
        f"{artifact_prefix(maze_name, observable, horizon)}_{agent.value}"
        f"{neural_context_suffix(agent, context_mode)}_run_dataset_{run_id}.npz"
    )


def checkpoint_label(
    agent: Agent,
    maze_name: str,
    observable: bool,
    context_mode: str | None = None,
    horizon: int | None = None,
) -> str:
    return (
        f"{artifact_prefix(maze_name, observable, horizon)}_{agent.value}"
        f"{neural_context_suffix(agent, context_mode)}_final"
    )


def evaluator_label(evaluator: Agent | EvaluatorIdentity) -> str:
    if isinstance(evaluator, Agent):
        return f"{evaluator.value}_fresh"
    if (
        is_neural_agent(evaluator.agent)
        and evaluator.context_mode is not None
        and evaluator.context_mode != DEFAULT_NEURAL_CONTEXT_MODE
    ):
        return f"{evaluator.agent.value}_{context_mode_artifact_label(evaluator.context_mode)}_{evaluator.mode.value}"
    return f"{evaluator.agent.value}_{evaluator.mode.value}"


def evaluator_label_for_agent(
    agent: Agent,
    mode: str | EvaluatorMode,
    context_mode: str | None = None,
) -> str:
    mode_value = EvaluatorMode(mode).value
    effective_context = resolve_agent_context_mode(agent, context_mode)
    if (
        effective_context is not None
        and effective_context != DEFAULT_NEURAL_CONTEXT_MODE
    ):
        return f"{agent.value}_{context_mode_artifact_label(effective_context)}_{mode_value}"
    return f"{agent.value}_{mode_value}"


def source_label(source: Agent, context_mode: str | None = None) -> str:
    return f"{source.value}{neural_context_suffix(source, context_mode)}"
