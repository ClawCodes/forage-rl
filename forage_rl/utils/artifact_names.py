"""Pure artifact naming helpers shared by file I/O code."""

from __future__ import annotations

import re
from pathlib import Path

from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    NeuralContextMode,
    canonical_agent,
    context_mode_token,
    is_neural_agent,
    legacy_alias_agents,
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


def neural_context_suffix(agent: Agent, context_mode: NeuralContextMode) -> str:
    if not is_neural_agent(agent) or context_mode == "legacy_context":
        return ""
    return f"_{context_mode_token(context_mode)}"


def load_candidate_agents(agent: Agent) -> tuple[Agent, ...]:
    resolved = canonical_agent(agent)
    return (resolved, *legacy_alias_agents(resolved))


def existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def extract_run_id(filepath: Path) -> int:
    match = re.search(r"_(\d+)\.npz$", filepath.name)
    if match is None:
        raise ValueError(f"Could not parse run id from filename: {filepath.name}")
    return int(match.group(1))


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
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> str:
    resolved = canonical_agent(agent)
    return (
        f"{artifact_prefix(maze_name, observable, horizon)}_{resolved.value}"
        f"{neural_context_suffix(resolved, context_mode)}_run_dataset_{run_id}.npz"
    )


def checkpoint_label(
    agent: Agent,
    maze_name: str,
    observable: bool,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> str:
    resolved = canonical_agent(agent)
    return (
        f"{artifact_prefix(maze_name, observable, horizon)}_{resolved.value}"
        f"{neural_context_suffix(resolved, context_mode)}_final"
    )


def evaluator_label(evaluator: Agent | EvaluatorSpec) -> str:
    if isinstance(evaluator, Agent):
        resolved = canonical_agent(evaluator)
        return f"{resolved.value}_fresh"
    resolved = canonical_agent(evaluator.agent)
    if is_neural_agent(resolved) and evaluator.context_mode != "legacy_context":
        return f"{resolved.value}_{context_mode_token(evaluator.context_mode)}_{evaluator.mode}"
    return f"{resolved.value}_{evaluator.mode}"


def evaluator_label_for_agent(
    agent: Agent,
    mode: str,
    context_mode: NeuralContextMode,
) -> str:
    if is_neural_agent(agent) and context_mode != "legacy_context":
        return f"{agent.value}_{context_mode_token(context_mode)}_{mode}"
    return f"{agent.value}_{mode}"


def source_label(source: Agent, context_mode: NeuralContextMode) -> str:
    resolved = canonical_agent(source)
    return f"{resolved.value}{neural_context_suffix(resolved, context_mode)}"


def source_label_for_agent(source: Agent, context_mode: NeuralContextMode) -> str:
    return f"{source.value}{neural_context_suffix(source, context_mode)}"
