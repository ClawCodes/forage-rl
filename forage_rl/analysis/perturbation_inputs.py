"""Helpers for externally supplied perturbation trajectories and metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from forage_rl.types import TimedTransition, Trajectory, Transition

T = TypeVar("T", bound=Transition)
_BENCHMARK_KINDS = {"true_mvt", "fo_oracle"}


def _transition_class(name: str) -> type[Transition]:
    normalized = name.strip()
    if normalized in {Transition.__name__, "transition"}:
        return Transition
    if normalized in {TimedTransition.__name__, "timed_transition"}:
        return TimedTransition
    raise ValueError(f"Unsupported transition_type {name!r}.")


def _bool_from_observable(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"fo", "true", "1", "fully_observed"}:
        return True
    if normalized in {"po", "false", "0", "partially_observed"}:
        return False
    raise ValueError(f"Unsupported observable value {value!r}.")


def _coerce_transition_list(
    raw_transitions: list[Any],
    transition_cls: type[T],
) -> list[T]:
    transitions: list[T] = []
    fields = list(transition_cls.model_fields.keys())
    for raw_transition in raw_transitions:
        if isinstance(raw_transition, dict):
            transitions.append(transition_cls.model_validate(raw_transition))
            continue
        if isinstance(raw_transition, (list, tuple)):
            transitions.append(
                transition_cls.model_validate(dict(zip(fields, raw_transition, strict=False)))
            )
            continue
        raise ValueError(
            "JSON trajectories must contain transition dicts or field-ordered lists, "
            f"got {type(raw_transition).__name__}."
        )
    return transitions


def load_combined_trajectory(
    trajectory_path: str | Path,
    *,
    transition_type: str,
) -> Trajectory:
    """Load one combined transition stream from `.npy`, `.npz`, or `.json`."""
    filepath = Path(trajectory_path).expanduser().resolve()
    transition_cls = _transition_class(transition_type)

    if filepath.suffix == ".npy":
        arr = np.load(filepath, allow_pickle=True)
        return Trajectory.from_numpy(np.asarray(arr), transition_cls)

    if filepath.suffix == ".npz":
        with np.load(filepath, allow_pickle=True) as payload:
            candidate_keys = [
                key for key in payload.files if key not in {"__transition_type__"}
            ]
            selected_key: str | None = None
            for key in ("transitions", "trajectory"):
                if key in candidate_keys:
                    selected_key = key
                    break
            if selected_key is None and len(candidate_keys) == 1:
                selected_key = candidate_keys[0]
            if selected_key is None:
                raise ValueError(
                    "Combined trajectory `.npz` files must expose exactly one transition "
                    "array or use a supported key like `transitions`."
                )
            return Trajectory.from_numpy(np.asarray(payload[selected_key]), transition_cls)

    if filepath.suffix == ".json":
        payload = json.loads(filepath.read_text(encoding="utf-8"))
        raw_transitions = payload["transitions"] if isinstance(payload, dict) else payload
        if not isinstance(raw_transitions, list):
            raise ValueError("JSON trajectory payload must be a list or contain `transitions`.")
        return Trajectory(
            transitions=_coerce_transition_list(raw_transitions, transition_cls)
        )

    raise ValueError(
        "Unsupported trajectory artifact format. Expected `.npy`, `.npz`, or `.json`, "
        f"got {filepath.suffix or '<no suffix>'}."
    )


def episode_trajectories_from_combined_stream(
    combined_stream: Trajectory[T] | list[T],
    episode_lengths: list[int] | tuple[int, ...],
) -> list[Trajectory[T]]:
    """Reconstruct episode trajectories from one ordered combined transition stream."""
    transitions = (
        list(combined_stream.transitions)
        if isinstance(combined_stream, Trajectory)
        else list(combined_stream)
    )
    lengths = [int(length) for length in episode_lengths]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"episode_lengths must all be > 0, got {lengths!r}.")
    if sum(lengths) != len(transitions):
        raise ValueError(
            "episode_lengths must sum to the combined transition count, "
            f"got sum={sum(lengths)} transitions={len(transitions)}."
        )

    trajectories: list[Trajectory[T]] = []
    offset = 0
    for length in lengths:
        trajectories.append(Trajectory(transitions=transitions[offset : offset + length]))
        offset += length
    return trajectories


def split_post_perturbation_episodes(
    combined_stream: Trajectory[T] | list[T],
    *,
    episode_lengths: list[int] | tuple[int, ...],
    perturbation_timestep: int,
    drop_mixed_episode: bool = True,
) -> tuple[list[int], list[Trajectory[T]]]:
    """Return fully post-perturbation episodes reconstructed from one combined stream."""
    episodes = episode_trajectories_from_combined_stream(combined_stream, episode_lengths)
    total_transitions = sum(int(length) for length in episode_lengths)
    if perturbation_timestep < 0 or perturbation_timestep > total_transitions:
        raise ValueError(
            "perturbation_timestep must lie within the combined transition range, "
            f"got {perturbation_timestep} for {total_transitions} transitions."
        )

    offset = 0
    post_start_index = len(episodes)
    for episode_index, trajectory in enumerate(episodes):
        start = offset
        end = start + len(trajectory)
        if start >= perturbation_timestep:
            post_start_index = episode_index
            break
        if start < perturbation_timestep < end:
            post_start_index = episode_index + 1 if drop_mixed_episode else episode_index
            break
        offset = end

    post_episode_indices = list(range(post_start_index, len(episodes)))
    return post_episode_indices, episodes[post_start_index:]


@dataclass(frozen=True)
class CombinedPerturbationRun:
    """Manifest-backed description of one externally produced perturbation run."""

    run_id: str
    agent: str
    maze_name: str
    observable: bool
    perturbation_kind: str
    perturbation_id: str
    trajectory_path: Path
    perturbation_timestep: int
    episode_lengths: tuple[int, ...]
    transition_type: str
    horizon: int
    context_mode: str | None = None
    benchmark_kind: str | None = None
    benchmark_params: dict[str, Any] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.perturbation_timestep < 0:
            raise ValueError(
                f"perturbation_timestep must be >= 0, got {self.perturbation_timestep}."
            )
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}.")
        if not self.episode_lengths:
            raise ValueError("episode_lengths must contain at least one episode.")
        if any(length <= 0 for length in self.episode_lengths):
            raise ValueError(
                f"episode_lengths must all be > 0, got {self.episode_lengths!r}."
            )
        _transition_class(self.transition_type)
        if self.benchmark_kind is not None and self.benchmark_kind not in _BENCHMARK_KINDS:
            raise ValueError(
                f"Unsupported benchmark_kind {self.benchmark_kind!r}. "
                f"Expected one of {sorted(_BENCHMARK_KINDS)}."
            )

    def load_combined_trajectory(self) -> Trajectory:
        """Load the externally provided combined transition stream."""
        return load_combined_trajectory(
            self.trajectory_path,
            transition_type=self.transition_type,
        )

    def episode_trajectories(self) -> list[Trajectory]:
        """Reconstruct all episode trajectories for this run."""
        return episode_trajectories_from_combined_stream(
            self.load_combined_trajectory(),
            self.episode_lengths,
        )

    def split_post_perturbation_episodes(
        self,
        *,
        drop_mixed_episode: bool = True,
    ) -> tuple[list[int], list[Trajectory]]:
        """Reconstruct fully post-perturbation episode trajectories for this run."""
        return split_post_perturbation_episodes(
            self.load_combined_trajectory(),
            episode_lengths=self.episode_lengths,
            perturbation_timestep=self.perturbation_timestep,
            drop_mixed_episode=drop_mixed_episode,
        )


def load_combined_perturbation_runs(
    manifest_path: str | Path,
) -> list[CombinedPerturbationRun]:
    """Load a JSON manifest containing one record per external perturbation run."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["runs"] if isinstance(payload, dict) and "runs" in payload else payload
    if not isinstance(records, list):
        raise ValueError("Manifest must be a JSON list or an object containing `runs`.")

    runs: list[CombinedPerturbationRun] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each manifest entry must be an object.")
        benchmark_params = record.get("benchmark_params")
        if benchmark_params is not None and not isinstance(benchmark_params, dict):
            raise ValueError("benchmark_params must be an object when provided.")
        if benchmark_params is not None and "maze_spec_path" in benchmark_params:
            benchmark_params = dict(benchmark_params)
            benchmark_params["maze_spec_path"] = str(
                (path.parent / benchmark_params["maze_spec_path"]).expanduser().resolve()
            )

        runs.append(
            CombinedPerturbationRun(
                run_id=str(record["run_id"]),
                agent=str(record["agent"]),
                maze_name=str(record["maze_name"]),
                observable=_bool_from_observable(record["observable"]),
                perturbation_kind=str(record["perturbation_kind"]),
                perturbation_id=str(record["perturbation_id"]),
                trajectory_path=(path.parent / record["trajectory_path"]).expanduser().resolve(),
                perturbation_timestep=int(record["perturbation_timestep"]),
                episode_lengths=tuple(int(length) for length in record["episode_lengths"]),
                transition_type=str(record["transition_type"]),
                horizon=int(record["horizon"]),
                context_mode=None
                if record.get("context_mode") is None
                else str(record["context_mode"]),
                benchmark_kind=None
                if record.get("benchmark_kind") is None
                else str(record["benchmark_kind"]),
                benchmark_params=benchmark_params,
                notes=None if record.get("notes") is None else str(record["notes"]),
            )
        )
    return runs
