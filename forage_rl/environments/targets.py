"""Helpers for selecting environment targets for experiments."""

from __future__ import annotations

from dataclasses import dataclass

from forage_rl.environments.maze import Maze, MazePOMDP
from forage_rl.environments.spec_loader import load_builtin_maze_spec


DEFAULT_ENV_TARGET = "simple:full"
SUPPORTED_ENV_TARGETS = ("simple:full", "full:full", "full:pomdp")


@dataclass(frozen=True)
class EnvironmentTarget:
    maze_name: str
    observability: str

    @property
    def key(self) -> str:
        return f"{self.maze_name}__{self.observability}"

    @property
    def spec_name(self) -> str:
        return self.maze_name

    @property
    def label(self) -> str:
        mode = "POMDP" if self.observability == "pomdp" else "full"
        return f"{self.maze_name} ({mode})"


def parse_environment_target(value: str) -> EnvironmentTarget:
    """Parse and validate a `maze:observability` target string."""
    normalized = value.strip().lower()
    if ":" not in normalized:
        raise ValueError(
            f"Environment targets must use 'maze:observability' syntax, got {value!r}."
        )

    maze_name, observability = normalized.split(":", 1)
    target = EnvironmentTarget(maze_name=maze_name, observability=observability)
    if target.observability == "pomdp" and target.maze_name == "simple":
        raise ValueError(
            f"{value!r} does not create meaningful partial observability. "
            "Use a maze with aliased observation groups."
        )
    if normalized not in SUPPORTED_ENV_TARGETS:
        raise ValueError(
            f"Unsupported environment target {value!r}. "
            f"Supported values: {', '.join(SUPPORTED_ENV_TARGETS)}"
        )

    if target.observability == "pomdp":
        spec = load_builtin_maze_spec(target.spec_name)
        if len({state.observation_group for state in spec.states}) == spec.num_states:
            raise ValueError(
                f"{value!r} does not create meaningful partial observability. "
                "Use a maze with aliased observation groups."
            )

    return target


def normalize_environment_targets(values: list[str] | None) -> list[EnvironmentTarget]:
    """Deduplicate and parse environment targets."""
    raw_values = [DEFAULT_ENV_TARGET] if not values else values
    ordered_unique = list(dict.fromkeys(raw_values))
    return [parse_environment_target(value) for value in ordered_unique]


def build_environment(target: EnvironmentTarget, seed: int | None = None):
    """Construct the environment described by an environment target."""
    spec = load_builtin_maze_spec(target.spec_name)
    if target.observability == "pomdp":
        return MazePOMDP(maze_spec=spec, seed=seed)
    return Maze(maze_spec=spec, seed=seed)


__all__ = [
    "DEFAULT_ENV_TARGET",
    "EnvironmentTarget",
    "SUPPORTED_ENV_TARGETS",
    "build_environment",
    "normalize_environment_targets",
    "parse_environment_target",
]
