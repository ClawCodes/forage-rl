"""File I/O utilities for saving and loading experiment data."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl.types import RunDataset, TimedTransition, Trajectory, Transition
from forage_rl.agents.registry import (
    Agent,
    EvaluatorSpec,
    NeuralContextMode,
    canonical_agent,
    is_neural_agent,
)
from forage_rl.config import (
    CHECKPOINTS_DIR,
    LOGPROBS_DIR,
    TRAJECTORIES_DIR,
    ensure_output_directories,
)
from forage_rl.environments import resolve_effective_horizon
from forage_rl.utils.artifact_names import (
    artifact_prefix,
    checkpoint_label,
    evaluator_label,
    evaluator_label_for_agent,
    existing_path,
    extract_run_id,
    is_canonical_run_dataset_file,
    load_candidate_agents,
    matches_exact_horizon_prefix,
    neural_context_suffix,
    obs_tag,
    run_dataset_filename,
    source_label,
    source_label_for_agent,
)


def _required_horizon(metadata: dict[str, object], *, artifact_label: str) -> int:
    if "horizon" not in metadata:
        raise ValueError(
            f"{artifact_label} is missing required horizon metadata and must be regenerated."
        )
    horizon = int(metadata["horizon"])
    if horizon <= 0:
        raise ValueError(f"{artifact_label} has invalid horizon metadata {horizon!r}.")
    return horizon


def _run_dataset_load_paths(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> list[Path]:
    return [
        TRAJECTORIES_DIR
        / (
            f"{artifact_prefix(maze_name, observable, horizon)}_{candidate.value}"
            f"{neural_context_suffix(candidate, context_mode)}"
            f"_run_dataset_{run_id}.npz"
        )
        for candidate in load_candidate_agents(agent)
    ]


def _run_dataset_path(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    return TRAJECTORIES_DIR / run_dataset_filename(
        agent,
        run_id,
        maze_name,
        observable,
        context_mode=context_mode,
        horizon=horizon,
    )


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


def _checkpoint_load_paths(
    agent: Agent,
    maze_name: str,
    observable: bool,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> list[Path]:
    return [
        CHECKPOINTS_DIR
        / (
            f"{artifact_prefix(maze_name, observable, horizon)}_{candidate.value}"
            f"{neural_context_suffix(candidate, context_mode)}_final.pt"
        )
        for candidate in load_candidate_agents(agent)
    ]


def checkpoint_path(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    """Return the canonical checkpoint path for a pretrained neural agent."""
    return CHECKPOINTS_DIR / (
        f"{
            checkpoint_label(
                agent,
                maze_name,
                observable,
                context_mode=context_mode,
                horizon=horizon,
            )
        }.pt"
    )


def checkpoint_metadata_path(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    """Return the canonical checkpoint metadata path."""
    return CHECKPOINTS_DIR / (
        f"{
            checkpoint_label(
                agent,
                maze_name,
                observable,
                context_mode=context_mode,
                horizon=horizon,
            )
        }.json"
    )


def resolve_checkpoint_load_path(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    """Resolve the first existing checkpoint path, falling back to legacy aliases."""
    return existing_path(
        _checkpoint_load_paths(
            agent,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=horizon,
        )
    )


def save_run_dataset(
    run_dataset: RunDataset,
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    """Save one run dataset as a `.npz` plus auxiliary JSON metadata."""
    ensure_output_directories()
    resolved_horizon = resolve_effective_horizon(maze_name, horizon)
    filepath = _run_dataset_path(
        agent,
        run_id,
        maze_name,
        observable,
        context_mode=context_mode,
        horizon=resolved_horizon,
    )
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
        "agent": canonical_agent(agent).value,
        "maze_name": maze_name,
        "observable": observable,
        "horizon": resolved_horizon,
        "num_episodes": run_dataset.num_episodes(),
        "num_transitions": run_dataset.num_transitions(),
    }
    if is_neural_agent(agent):
        metadata["context_mode"] = context_mode
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return filepath


def load_run_dataset(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> RunDataset:
    """Load one run dataset from the current `.npz` format."""
    filepath = existing_path(
        _run_dataset_load_paths(
            agent,
            run_id,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=horizon,
        )
    )
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


def load_run_dataset_metadata(
    agent: Agent,
    run_id: int,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> dict[str, object]:
    """Load auxiliary metadata for one saved run dataset."""
    filepath = existing_path(
        _run_dataset_load_paths(
            agent,
            run_id,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=horizon,
        )
    )
    metadata_path = _run_dataset_metadata_path(filepath)
    if not metadata_path.exists():
        raise ValueError(
            f"Run dataset metadata is missing for {filepath.name}; regenerate this artifact."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _required_horizon(metadata, artifact_label=f"Run dataset {filepath.name}")
    return metadata


def load_checkpoint_metadata(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> dict[str, object]:
    """Load auxiliary metadata for one saved checkpoint."""
    metadata_path = checkpoint_metadata_path(
        agent,
        maze_name,
        observable,
        context_mode=context_mode,
        horizon=horizon,
    )
    if not metadata_path.exists():
        raise ValueError(
            f"Checkpoint metadata is missing for {metadata_path.name}; regenerate this artifact."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _required_horizon(metadata, artifact_label=f"Checkpoint {metadata_path.name}")
    return metadata


def save_logprobs(
    data: np.ndarray,
    source: Agent,
    evaluator: Agent | EvaluatorSpec,
    run_id: int,
    maze_name: str,
    observable: bool = True,
    source_context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> Path:
    """Save log probability data to organized directory."""
    ensure_output_directories()
    label = (
        f"source_{source_label(source, source_context_mode)}_eval_"
        f"{evaluator_label(evaluator)}"
    )
    filename = (
        f"{artifact_prefix(maze_name, observable, horizon)}_{label}"
        f"_log_likelihoods_{run_id}.npy"
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
    source_context_mode: NeuralContextMode = "legacy_context",
    evaluator_context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> np.ndarray:
    """Load log probability data from organized directory."""
    source_candidates = load_candidate_agents(source)
    if isinstance(evaluator, Agent):
        evaluator_mode_value = evaluator_mode
        evaluator_context = evaluator_context_mode
        evaluator_candidates = load_candidate_agents(evaluator)
    else:
        evaluator_mode_value = evaluator.mode
        evaluator_context = evaluator.context_mode
        evaluator_candidates = load_candidate_agents(evaluator.agent)

    candidate_paths: list[Path] = []
    for source_candidate in source_candidates:
        for evaluator_candidate in evaluator_candidates:
            label = (
                f"source_{source_label_for_agent(source_candidate, source_context_mode)}"
                f"_eval_{evaluator_label_for_agent(evaluator_candidate, evaluator_mode_value, evaluator_context)}"
            )
            filename = (
                f"{artifact_prefix(maze_name, observable, horizon)}_{label}"
                f"_log_likelihoods_{run_id}.npy"
            )
            candidate_paths.append(LOGPROBS_DIR / filename)

    filepath = existing_path(candidate_paths)
    return np.load(filepath, allow_pickle=True)


def list_run_dataset_files(
    agent: Optional[Agent] = None,
    maze_name: Optional[str] = None,
    observable: Optional[bool] = None,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> list[Path]:
    """List all saved run-dataset files in the data directory."""
    if not TRAJECTORIES_DIR.exists():
        return []
    if horizon is not None and (maze_name is None or observable is None):
        raise ValueError(
            "maze_name and observable are required when filtering run datasets by horizon."
        )

    exact_prefix = maze_name is not None and observable is not None
    if exact_prefix:
        prefix = artifact_prefix(maze_name, observable, horizon)
    else:
        maze_part = maze_name or "*"
        obs_part = obs_tag(observable) if observable is not None else "*"
        prefix = f"{maze_part}_{obs_part}"
    if agent is None:
        agent_part = "*"
        return sorted(
            path
            for path in TRAJECTORIES_DIR.glob(
                f"{prefix}_{agent_part}*_run_dataset_*.npz"
            )
            if is_canonical_run_dataset_file(path)
            if (not exact_prefix)
            or matches_exact_horizon_prefix(path, prefix, horizon=horizon)
        )

    paths: set[Path] = set()
    for candidate in load_candidate_agents(agent):
        agent_part = (
            f"{candidate.value}{neural_context_suffix(candidate, context_mode)}"
        )
        pattern = f"{prefix}_{agent_part}*_run_dataset_*.npz"
        paths.update(
            path
            for path in TRAJECTORIES_DIR.glob(pattern)
            if is_canonical_run_dataset_file(path)
            if (not exact_prefix)
            or matches_exact_horizon_prefix(path, prefix, horizon=horizon)
        )
    return sorted(paths)


def list_run_dataset_run_ids(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> list[int]:
    """Return sorted saved run ids for an agent/maze combination."""
    return sorted(
        {
            extract_run_id(filepath)
            for filepath in list_run_dataset_files(
                agent,
                maze_name,
                observable,
                context_mode=context_mode,
                horizon=horizon,
            )
        }
    )


# Re-export the extracted naming helpers to keep the current private module API stable.
_obs_tag = obs_tag
_artifact_prefix = artifact_prefix
_neural_context_suffix = neural_context_suffix
_load_candidate_agents = load_candidate_agents
_existing_path = existing_path
_extract_run_id = extract_run_id
_is_canonical_run_dataset_file = is_canonical_run_dataset_file
_matches_exact_horizon_prefix = matches_exact_horizon_prefix
_run_dataset_filename = run_dataset_filename
_checkpoint_label = checkpoint_label
_evaluator_label = evaluator_label
_evaluator_label_for_agent = evaluator_label_for_agent
_source_label = source_label
_source_label_for_agent = source_label_for_agent


def get_run_count(
    agent: Agent,
    maze_name: str,
    observable: bool = True,
    context_mode: NeuralContextMode = "legacy_context",
    horizon: int | None = None,
) -> int:
    """Get the number of saved run datasets for an agent and maze."""
    return len(
        list_run_dataset_run_ids(
            agent,
            maze_name,
            observable,
            context_mode=context_mode,
            horizon=horizon,
        )
    )
