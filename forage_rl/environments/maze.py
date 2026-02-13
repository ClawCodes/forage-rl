"""Foraging maze environments driven by validated maze specifications."""

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from forage_rl.types import Transition

from .spec_loader import load_builtin_maze_spec, load_maze_spec
from .specs import (
    MazeMeta,
    MazeSpec,
    StateSpec,
    TransitionDurationSpec,
    TransitionStepSpec,
)
from ..config import DefaultParams


class ForagingReward:
    """Models time-dependent reward depletion in a foraging patch."""

    def __init__(self, decay: float, rng: np.random.Generator):
        """Initialize decay process with a dedicated RNG stream."""
        self.decay = decay
        self.counter = 0
        self.rng = rng

    def reset(self):
        """Reset the depletion counter when leaving a patch."""
        self.counter = 0

    def sample_reward(self) -> float:
        """Sample reward as Bernoulli(exp(-decay * counter))."""
        reward_prob = np.exp(-self.decay * self.counter)
        self.counter += 1
        return 1.0 if self.rng.random() < reward_prob else 0.0


class Maze(gym.Env):
    """
    Maze environment with transitions and rewards defined by a TOML spec.

    Implements the standard Gymnasium interface (step returns a 5-tuple)
    and provides ``step_transition()`` for backward compatibility with
    existing agents that expect ``(Transition, bool)``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        maze_spec: MazeSpec,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> None:
        """Initialize maze dynamics from a validated spec or TOML file."""
        super().__init__()

        self.maze_spec = maze_spec
        self.horizon = horizon or self.maze_spec.maze.horizon
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")

        self.rng = rng or np.random.default_rng(seed)
        self.num_states = self.maze_spec.num_states
        self.num_actions = self.maze_spec.num_actions
        self.decays = self.maze_spec.decays
        self.state_labels = self.maze_spec.state_labels
        self.action_labels = list(self.maze_spec.maze.action_labels)
        self.initial_state = self.maze_spec.maze.initial_state

        # Gymnasium spaces
        self.observation_space = Discrete(self.num_states)
        self.action_space = Discrete(self.num_actions)

        # Internal state
        self.state = self.initial_state
        self.time = 0

        # Precomputed transition tables
        self._transitions_by_state_action = self.maze_spec.transition_map()
        self._transition_duration_by_edge: dict[tuple[int, int, int], int] = {}
        if self.maze_spec.uses_transition_durations:
            self._transition_duration_by_edge = {
                (row.state, row.action, row.next_state): row.duration
                for row in self.maze_spec.transitions
                if isinstance(row, TransitionDurationSpec)
            }

        self.reward_models = [ForagingReward(decay, self.rng) for decay in self.decays]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> "Maze":
        """Build a maze from a TOML specification file path."""
        spec = load_maze_spec(path)
        return cls(maze_spec=spec, seed=seed, rng=rng, horizon=horizon)

    @classmethod
    def from_spec(
        cls,
        name: str = "simple",
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> "Maze":
        """Build a maze from an existing TOML specification file name."""
        spec = load_builtin_maze_spec(name)
        return cls(maze_spec=spec, seed=seed, rng=rng, horizon=horizon)

    def _validate_state(self, state_idx: int):
        if state_idx < 0 or state_idx >= self.num_states:
            raise ValueError(
                f"state {state_idx} is out of range [0, {self.num_states})"
            )

    def _validate_action(self, action_idx: int):
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(
                f"action {action_idx} is out of range [0, {self.num_actions})"
            )

    def get_state_label(self, state_idx: int) -> str:
        """Return human-readable label for a state index."""
        self._validate_state(state_idx)
        return self.state_labels[state_idx]

    def get_action_label(self, action_idx: int) -> str:
        """Return human-readable label for an action index."""
        self._validate_action(action_idx)
        return self.action_labels[action_idx]

    def transition_distribution(
        self, state_idx: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Return sorted (next_state, probability) transitions for a state-action."""
        self._validate_state(state_idx)
        self._validate_action(action_idx)
        return self._transitions_by_state_action[(state_idx, action_idx)]

    def _sample_next_state(self, state_idx: int, action_idx: int) -> int:
        transition_distribution = self.transition_distribution(state_idx, action_idx)
        next_state_ids, transition_probs = zip(*transition_distribution)
        return int(self.rng.choice(next_state_ids, p=transition_probs))

    def transition_duration(
        self, state_idx: int, action_idx: int, next_state_idx: int
    ) -> int:
        """Return time cost for a concrete ``(state, action, next_state)`` edge."""
        if not self.maze_spec.uses_transition_durations:
            return 1

        transition_key = (state_idx, action_idx, next_state_idx)
        try:
            return self._transition_duration_by_edge[transition_key]
        except KeyError as exc:
            raise ValueError(
                "No transition duration defined for "
                f"state={state_idx}, action={action_idx}, next_state={next_state_idx}"
            ) from exc

    def _get_reward(self, next_state_idx: int) -> float:
        """Return reward for a transition and reset depletion after patch switches."""
        if self.state == next_state_idx:
            return self.reward_models[self.state].sample_reward()

        for reward_model in self.reward_models:
            reward_model.reset()
        return 0.0

    # --- Begin Gymnasium interface ---
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset to initial state and clear per-patch depletion counters."""
        super().reset(seed=seed, options=options)
        self.state = self.initial_state
        self.time = 0
        for reward_model in self.reward_models:
            reward_model.reset()
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Execute action and return standard Gymnasium 5-tuple.

        Returns:
            (obs, reward, terminated, truncated, info) where
            ``terminated`` is always False (no absorbing states) and
            ``truncated`` is True when ``time >= horizon``.
        """
        self._validate_action(action)
        prev_state = self.state
        next_state = self._sample_next_state(prev_state, action)
        duration = self.transition_duration(prev_state, action, next_state)
        reward = self._get_reward(next_state)

        self.state = next_state
        self.time += duration
        truncated = self.time >= self.horizon

        info = {"prev_state": prev_state, "action": action}
        return self.state, reward, False, truncated, info

    # Backward compatible interface
    def step_transition(self, action: int) -> tuple[Transition, bool]:
        """Execute action and return ``(Transition, done)``.

        Convenience wrapper around ``step()`` for agents that expect the
        legacy ``(Transition, bool)`` interface.
        """
        obs, reward, terminated, truncated, info = self.step(action)
        transition = Transition(
            state=info["prev_state"],
            action=info["action"],
            reward=reward,
            next_state=obs,
        )
        done = terminated or truncated
        return transition, done


class MazePOMDP(Maze):
    """Partially observable wrapper with explicit observation groups from spec."""

    def __init__(
        self,
        maze_spec: MazeSpec,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ):
        """Initialize POMDP wrapper with explicit observation-group mapping."""
        super().__init__(
            maze_spec=maze_spec,
            seed=seed,
            rng=rng,
            horizon=horizon,
        )
        self._state_to_observation_group = self._build_state_observation_map()
        self.num_observations = len(set(self._state_to_observation_group.values()))
        self.observation_space = Discrete(self.num_observations)

    def _build_state_observation_map(self) -> dict[int, int]:
        """Build mapping from concrete states to observation groups."""

        return {state.id: state.observation_group for state in self.maze_spec.states}

    def observe(self, state: Optional[int] = None) -> int:
        """Return observation group id for a state (or current state by default)."""
        state_idx = state or self.state
        self._validate_state(state_idx)
        return self._state_to_observation_group[state_idx]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset and return observation group instead of true state."""
        _, info = super().reset(seed=seed, options=options)
        info["true_state"] = self.state
        return self.observe(), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Step and return observation group instead of true state."""
        _, reward, terminated, truncated, info = super().step(action)
        info["true_state"] = self.state
        return self.observe(), reward, terminated, truncated, info


def _simple_spec_from_decays(
    decays: list[float], horizon: int = DefaultParams.HORIZON
) -> MazeSpec:
    """Construct a simple 2-state spec from custom decay values."""
    if len(decays) != 2:
        raise ValueError(f"SimpleMaze expects 2 decays, got {len(decays)}")

    return MazeSpec(
        maze=MazeMeta(
            name="simple-custom",
            horizon=horizon,
            initial_state=0,
            action_labels=["stay", "leave"],
        ),
        states=[
            StateSpec(
                id=0,
                label="Upper Patch",
                decay=decays[0],
                observation_group=0,
            ),
            StateSpec(
                id=1,
                label="Lower Patch",
                decay=decays[1],
                observation_group=1,
            ),
        ],
        transitions=[
            TransitionStepSpec(state=0, action=0, next_state=0, prob=1.0),
            TransitionStepSpec(state=0, action=1, next_state=1, prob=1.0),
            TransitionStepSpec(state=1, action=0, next_state=1, prob=1.0),
            TransitionStepSpec(state=1, action=1, next_state=0, prob=1.0),
        ],
    )


class SimpleMaze(Maze):
    """Compatibility wrapper for the default simple maze variant."""

    def __init__(
        self,
        decays: Optional[list[float]] = None,
        horizon: int = DefaultParams.HORIZON,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize compatibility wrapper for default or custom simple specs."""
        if decays is not None:
            super().__init__(
                maze_spec=_simple_spec_from_decays(decays=decays, horizon=horizon),
                seed=seed,
                rng=rng,
                horizon=horizon,
            )
            return

        super().__init__(
            maze_spec=load_builtin_maze_spec("simple"),
            seed=seed,
            rng=rng,
            horizon=horizon,
        )
