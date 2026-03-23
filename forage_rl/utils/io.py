"""File I/O utilities for saving and loading experiment data."""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl import RunDataset, TimedTransition, Trajectory, Transition
from forage_rl.agents.registry import Agent, EvaluatorSpec
from forage_rl.config import (
    CHECKPOINTS_DIR,
    LOGPROBS_DIR,
    TRAJECTORIES_DIR,
    ensure_directories,
)


def _obs_tag(observable: bool) -> str:
    return "FO" if observable else "PO"


def _extract_run_id(filepath: Path) -> int:
    match = re.search(r"_(\d+)\.npz$", filepath.name)
    if match is None:
        raise ValueError(f"Could not parse run id from filename: {filepath.name}")
    return int(match.group(1))


def _run_dataset_filename(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool,
) -> str:
    return (
        f"{maze_name}_{_obs_tag(observable)}_{agent.value}_run_dataset_{run_id}.npz"
    )


def _run_dataset_path(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool,
) -> Path:
    return TRAJECTORIES_DIR / _run_dataset_filename(agent, run_id, maze_name, observable)


def _run_dataset_metadata_path(filepath: Path) -> Path:
    return filepath.with_suffix(".json")


def _transition_name(transition_cls: type[Transition]) -> str:
    return transition_cls.__name__


def _transition_class(name: str) -> type[Transition]:
    if name == Transition.__name__:
        return Transition
    if name == TimedTransition.__name__:
        return TimedTransition
    raise ValueError(f"Unsupported transition type {name!r}.")


def _checkpoint_label(agent: Agent, maze_name: str, observable: bool) -> str:
    return f"{maze_name}_{_obs_tag(observable)}_{agent.value}_final"


def checkpoint_path(agent: Agent, maze_name: str, observable: bool = True) -> Path:
    """Return the canonical checkpoint path for a pretrained neural agent."""
    return CHECKPOINTS_DIR / f"{_checkpoint_label(agent, maze_name, observable)}.pt"


def checkpoint_metadata_path(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
) -> Path:
    """Return the canonical checkpoint metadata path."""
    return CHECKPOINTS_DIR / f"{_checkpoint_label(agent, maze_name, observable)}.json"


def _evaluator_label(evaluator: Agent | EvaluatorSpec) -> str:
    if isinstance(evaluator, Agent):
        return f"{evaluator.value}_fresh"
    return evaluator.label


def save_run_dataset(
    run_dataset: RunDataset,
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> Path:
    """Save one run dataset as a `.npz` plus auxiliary JSON metadata."""
    ensure_directories()
    filepath = _run_dataset_path(agent, run_id, maze_name, observable)
    payload: dict[str, object] = {
        "__transition_type__": np.array(
            _transition_name(run_dataset.transition_cls()),
            dtype=np.str_,
        ),
    }
    for episode_index, trajectory in enumerate(run_dataset.trajectories):
        payload[f"episode_{episode_index:05d}"] = trajectory.to_numpy()
    np.savez(filepath, **payload)

    metadata_path = _run_dataset_metadata_path(filepath)
    metadata = {
        "container_type": "run_dataset",
        "agent": agent.value,
        "maze_name": maze_name,
        "observable": observable,
        "num_episodes": run_dataset.num_episodes(),
        "num_transitions": run_dataset.num_transitions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return filepath


def load_run_dataset(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> RunDataset:
    """Load one run dataset from the current `.npz` format."""
    filepath = _run_dataset_path(agent, run_id, maze_name, observable)
    with np.load(filepath, allow_pickle=True) as payload:
        transition_name = str(payload["__transition_type__"].item())
        transition_cls = _transition_class(transition_name)
        episode_keys = sorted(
            key for key in payload.files if key.startswith("episode_")
        )
        trajectories = [
            Trajectory.from_numpy(payload[episode_key], transition_cls)
            for episode_key in episode_keys
        ]
    return RunDataset(trajectories=trajectories)


def save_logprobs(
    data: np.ndarray,
    source: Agent,
    evaluator: Agent | EvaluatorSpec,
    run_id: int,
    maze_name: str,
    observable: bool = True,
) -> Path:
    """Save log probability data to organized directory."""
    ensure_directories()
    label = f"source_{source.value}_eval_{_evaluator_label(evaluator)}"
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{label}_log_likelihoods_{run_id}.npy"
    )
    filepath = LOGPROBS_DIR / filename
    np.save(filepath, data)
    return filepath


def load_logprobs(
    source: Agent,
    evaluator: Agent | EvaluatorSpec,
    run_id: int,
    maze_name: str,
    observable: bool = True,
    evaluator_mode: str = "fresh",
) -> np.ndarray:
    """Load log probability data from organized directory."""
    if isinstance(evaluator, Agent):
        label = f"source_{source.value}_eval_{evaluator.value}_{evaluator_mode}"
        filename = (
            f"{maze_name}_{_obs_tag(observable)}_{label}_log_likelihoods_{run_id}.npy"
        )
        filepath = LOGPROBS_DIR / filename
        return np.load(filepath, allow_pickle=True)

    label = f"source_{source.value}_eval_{evaluator.label}"
    filename = (
        f"{maze_name}_{_obs_tag(observable)}_{label}_log_likelihoods_{run_id}.npy"
    )
    filepath = LOGPROBS_DIR / filename
    return np.load(filepath, allow_pickle=True)


def list_run_dataset_files(
    agent: Optional[Agent] = None,
    maze_name: Optional[str] = None,
    observable: Optional[bool] = None,
) -> list[Path]:
    """List all saved run-dataset files in the data directory."""
    if not TRAJECTORIES_DIR.exists():
        return []

    maze_part = maze_name or "*"
    obs_part = _obs_tag(observable) if observable is not None else "*"
    agent_part = agent.value if agent is not None else "*"
    pattern = f"{maze_part}_{obs_part}_{agent_part}_run_dataset_*.npz"
    return sorted(TRAJECTORIES_DIR.glob(pattern))


def list_run_dataset_run_ids(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
) -> list[int]:
    """Return sorted saved run ids for an agent/maze combination."""
    return sorted(
        _extract_run_id(filepath)
        for filepath in list_run_dataset_files(agent, maze_name, observable)
    )


def list_logprob_files(
    label: Optional[str] = None,
    maze_name: Optional[str] = None,
    observable: Optional[bool] = None,
) -> list[Path]:
    """List all log probability files in the data directory."""
    if not LOGPROBS_DIR.exists():
        return []

    maze_part = maze_name or "*"
    obs_part = _obs_tag(observable) if observable is not None else "*"
    label_part = label or "*"
    pattern = f"{maze_part}_{obs_part}_{label_part}_log_likelihoods_*.npy"
    return sorted(LOGPROBS_DIR.glob(pattern))


def get_run_count(agent: Agent, maze_name: str, observable: bool = True) -> int:
    """Get the number of saved run datasets for an agent and maze."""
    return len(list_run_dataset_run_ids(agent, maze_name, observable))
