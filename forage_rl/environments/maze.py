"""Foraging maze environments driven by validated maze specifications."""

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from forage_rl.types import Transition

from .spec_loader import load_builtin_maze_spec, load_maze_spec
from .specs import MazeSpec, TransitionDurationSpec


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
    """Maze environment with transitions and rewards defined by a TOML spec."""

    def __init__(
        self,
        spec: Optional[MazeSpec] = None,
        spec_path: Optional[str | Path] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> None:
        """Initialize maze dynamics from a validated spec or TOML file."""
        if spec is not None and spec_path is not None:
            raise ValueError("Provide only one of spec or spec_path")

        if spec_path is not None:
            resolved_spec = load_maze_spec(spec_path)
        elif spec is not None:
            resolved_spec = spec
        else:
            resolved_spec = load_builtin_maze_spec("simple")

        self.spec = resolved_spec
        self.horizon = horizon if horizon is not None else self.spec.maze.horizon
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.num_states = self.spec.num_states
        self.num_actions = self.spec.num_actions
        self.decays = self.spec.decays
        self.state_labels = self.spec.state_labels
        self.action_labels = list(self.spec.maze.action_labels)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.initial_state = self.spec.maze.initial_state
        self.state = self.initial_state
        self.time = 0
        self._transitions_by_state_action = self.spec.transition_map()
        self._transition_duration_by_edge: dict[tuple[int, int, int], int] = {}
        if self.spec.uses_transition_durations:
            self._transition_duration_by_edge = {
                (row.state, row.action, row.next_state): row.duration
                for row in self.spec.transitions
                if isinstance(row, TransitionDurationSpec)
        }
        self.reward_models = [ForagingReward(decay, self.rng) for decay in self.decays]

    @classmethod
    def from_spec(
        cls,
        spec: MazeSpec,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> "Maze":
        """Build a maze directly from an in-memory `MazeSpec`."""
        return cls(spec=spec, seed=seed, rng=rng, horizon=horizon)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> "Maze":
        """Build a maze from a TOML specification file path."""
        return cls(spec_path=path, seed=seed, rng=rng, horizon=horizon)

    @classmethod
    def default(
        cls,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ) -> "Maze":
        """Build the default bundled 2-state simple maze."""
        return cls(spec=load_builtin_maze_spec("simple"), seed=seed, rng=rng, horizon=horizon)

    def _validate_state(self, state_idx: int):
        """Validate state index bounds."""
        if state_idx < 0 or state_idx >= self.num_states:
            raise ValueError(
                f"state {state_idx} is out of range [0, {self.num_states})"
            )

    def _validate_action(self, action_idx: int):
        """Validate action index bounds."""
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

    def _init_reward_models(self) -> None:
        self.reward_models = [ForagingReward(decay, self.rng) for decay in self.decays]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset to initial state and clear per-patch depletion counters."""
        super().reset(seed=seed)
        _ = options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._init_reward_models()
        self.state = self.initial_state
        self.time = 0
        for reward_model in self.reward_models:
            reward_model.reset()
        return self.state, {"state": self.state, "time": self.time}

    def transition_distribution(
        self, state_idx: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Return sorted (next_state, probability) transitions for a state-action."""
        self._validate_state(state_idx)
        self._validate_action(action_idx)
        return list(self._transitions_by_state_action[(state_idx, action_idx)])

    def _sample_next_state(self, state_idx: int, action_idx: int) -> int:
        transition_distribution = self.transition_distribution(state_idx, action_idx)
        next_state_ids, transition_probs = zip(*transition_distribution)
        return int(self.rng.choice(next_state_ids, p=transition_probs))

    def transition_duration(
        self, state_idx: int, action_idx: int, next_state_idx: int
    ) -> int:
        """Return time cost for a concrete `(state, action, next_state)` edge."""
        if not self.spec.uses_transition_durations:
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

        # Leaving a patch resets all depletion counters; travel gets no reward.
        for reward_model in self.reward_models:
            reward_model.reset()
        return 0.0

    def _make_transition(
        self,
        prev_state_idx: int,
        action_idx: int,
        reward_value: float,
        next_state_idx: int,
    ) -> Transition:
        return Transition(
            state=prev_state_idx,
            action=action_idx,
            reward=reward_value,
            next_state=next_state_idx,
        )

    def _build_info(
        self,
        prev_state_idx: int,
        action_idx: int,
        next_state_idx: int,
        transition_duration: int,
        reward_value: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        transition = self._make_transition(
            prev_state_idx=prev_state_idx,
            action_idx=action_idx,
            reward_value=reward_value,
            next_state_idx=next_state_idx,
        )
        return {
            "state": prev_state_idx,
            "action": action_idx,
            "reward": reward_value,
            "next_state": next_state_idx,
            "time": self.time,
            "transition_duration": transition_duration,
            "terminated": terminated,
            "truncated": truncated,
            "transition": transition,
        }

    def step(
        self, action_idx: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Execute action and return Gymnasium step tuple."""
        self._validate_action(action_idx)
        prev_state_idx = self.state
        next_state_idx = self._sample_next_state(prev_state_idx, action_idx)
        transition_duration = self.transition_duration(
            prev_state_idx, action_idx, next_state_idx
        )
        reward_value = self._get_reward(next_state_idx)
        self.state = next_state_idx
        self.time += transition_duration
        terminated = False
        truncated = self.time >= self.horizon
        info = self._build_info(
            prev_state_idx=prev_state_idx,
            action_idx=action_idx,
            next_state_idx=next_state_idx,
            transition_duration=transition_duration,
            reward_value=reward_value,
            terminated=terminated,
            truncated=truncated,
        )
        return next_state_idx, reward_value, terminated, truncated, info


class MazePOMDP(Maze):
    """Partially observable wrapper with explicit observation groups from spec."""

    def __init__(
        self,
        spec: Optional[MazeSpec] = None,
        spec_path: Optional[str | Path] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        horizon: Optional[int] = None,
    ):
        """Initialize POMDP wrapper with explicit observation-group mapping."""
        super().__init__(
            spec=spec,
            spec_path=spec_path,
            seed=seed,
            rng=rng,
            horizon=horizon,
        )
        self._state_to_observation_group = self._build_state_observation_map()
        self.num_observations = len(set(self._state_to_observation_group.values()))
        self.observation_space = spaces.Discrete(self.num_observations)

    def _build_state_observation_map(self) -> dict[int, int]:
        """Build mapping from concrete states to observation groups."""
        mapping: dict[int, int] = {}
        for state in sorted(self.spec.states, key=lambda s: s.id):
            mapping[state.id] = state.observation_group
        return mapping

    def observe(self, state: Optional[int] = None) -> int:
        """Return observation group id for a state (or current state by default)."""
        state_idx = self.state if state is None else state
        self._validate_state(state_idx)
        return self._state_to_observation_group[state_idx]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        state_idx, info = super().reset(seed=seed, options=options)
        observation = self.observe(state_idx)
        info["observation"] = observation
        return observation, info

    def step(
        self, action_idx: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        _, reward_value, terminated, truncated, info = super().step(action_idx)
        next_state_idx = int(info["next_state"])
        observation = self.observe(next_state_idx)
        info["observation"] = observation
        return observation, reward_value, terminated, truncated, info
