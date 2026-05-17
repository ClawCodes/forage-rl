"""Benchmark selection and reconstruction for perturbation recovery analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from forage_rl.analysis.mvt import (
    simple_one_way_true_mvt_optimal_prt,
    simple_true_mvt_optimal_prt,
)
from forage_rl.analysis.oracle_patch_benchmark import oracle_patch_optimal_prt_by_state
from forage_rl.environments import Maze, MazeSpec, load_builtin_maze_spec, load_maze_spec

BenchmarkKind = Literal["true_mvt", "fo_oracle"]
BenchmarkMode = Literal["auto", "true_mvt", "fo_oracle"]
_BENCHMARK_KINDS = {"true_mvt", "fo_oracle"}
_BENCHMARK_MODES = {"auto", "true_mvt", "fo_oracle"}


def resolve_patch_benchmark_kind(
    maze_name: str,
    observable: bool,
) -> BenchmarkKind:
    """Resolve the honest benchmark family for a maze/observability condition."""
    del observable
    if maze_name in {"simple", "simple_one_way"}:
        return "true_mvt"
    if maze_name in {"full", "full_one_way"}:
        return "fo_oracle"
    raise ValueError(f"Unsupported maze_name {maze_name!r} for patch benchmark resolution.")


def _apply_state_overrides(
    spec_data: dict[str, Any],
    state_overrides: Mapping[str | int, Mapping[str, Any]],
) -> dict[str, Any]:
    states_by_id = {
        int(state_data["id"]): dict(state_data) for state_data in spec_data["states"]
    }
    for state_id, overrides in state_overrides.items():
        resolved_state_id = int(state_id)
        if resolved_state_id not in states_by_id:
            raise ValueError(f"Unknown state override id {resolved_state_id}.")
        states_by_id[resolved_state_id].update(dict(overrides))
    spec_data["states"] = [
        states_by_id[state_id] for state_id in sorted(states_by_id)
    ]
    return spec_data


def _build_benchmark_maze(
    *,
    maze_name: str,
    horizon: int,
    benchmark_params: Mapping[str, Any] | None,
) -> Maze:
    if benchmark_params is None:
        return Maze(load_builtin_maze_spec(maze_name), seed=0, horizon=horizon)

    if "maze_spec_path" in benchmark_params:
        maze_spec = load_maze_spec(benchmark_params["maze_spec_path"])
    else:
        maze_spec = load_builtin_maze_spec(maze_name)

    spec_data = maze_spec.model_dump(mode="python")
    state_overrides = benchmark_params.get("state_overrides", benchmark_params.get("states"))
    if state_overrides is not None:
        if not isinstance(state_overrides, Mapping):
            raise ValueError("state_overrides must be a mapping of state ids to overrides.")
        spec_data = _apply_state_overrides(spec_data, state_overrides)

    if "transitions" in benchmark_params:
        spec_data["transitions"] = benchmark_params["transitions"]
    if "maze" in benchmark_params:
        if not isinstance(benchmark_params["maze"], Mapping):
            raise ValueError("benchmark_params['maze'] must be an object.")
        spec_data["maze"] = {**spec_data["maze"], **dict(benchmark_params["maze"])}

    updated_spec = MazeSpec.model_validate(spec_data)
    return Maze(updated_spec, seed=0, horizon=horizon)


def build_patch_benchmark_maze(
    *,
    maze_name: str,
    horizon: int,
    benchmark_params: Mapping[str, Any] | None = None,
) -> Maze:
    """Build the fully observed perturbed maze used for benchmark reconstruction."""
    return _build_benchmark_maze(
        maze_name=maze_name,
        horizon=horizon,
        benchmark_params=benchmark_params,
    )


def resolve_patch_benchmark_prt(
    *,
    maze_name: str,
    observable: bool,
    horizon: int,
    benchmark_mode: BenchmarkMode = "auto",
    benchmark_params: Mapping[str, Any] | None = None,
) -> dict[int, int]:
    """Resolve benchmark patch residence times for one perturbed maze condition."""
    if benchmark_mode not in _BENCHMARK_MODES:
        raise ValueError(
            f"Unsupported benchmark_mode {benchmark_mode!r}. "
            f"Expected one of {sorted(_BENCHMARK_MODES)}."
        )

    benchmark_kind: BenchmarkKind
    if benchmark_mode == "auto":
        benchmark_kind = resolve_patch_benchmark_kind(maze_name, observable)
    else:
        benchmark_kind = benchmark_mode

    if benchmark_kind not in _BENCHMARK_KINDS:
        raise ValueError(
            f"Unsupported benchmark_kind {benchmark_kind!r}. "
            f"Expected one of {sorted(_BENCHMARK_KINDS)}."
        )

    benchmark_maze = _build_benchmark_maze(
        maze_name=maze_name,
        horizon=horizon,
        benchmark_params=benchmark_params,
    )

    if benchmark_kind == "true_mvt":
        if maze_name == "simple":
            return simple_true_mvt_optimal_prt(benchmark_maze, horizon)
        if maze_name == "simple_one_way":
            return simple_one_way_true_mvt_optimal_prt(benchmark_maze, horizon)
        raise ValueError(
            "True MVT benchmarks are only exact for `simple` and `simple_one_way`, "
            f"got {maze_name!r}."
        )

    return oracle_patch_optimal_prt_by_state(benchmark_maze)
