"""Foraging maze environments driven by validated maze specifications."""

from collections import defaultdict
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

    def __init__(
        self,
        *,
        decay: float | None,
        initial_reward_prob: float = 1.0,
        reward_probs: list[float] | None = None,
        rng: np.random.Generator,
    ):
        """Initialize decay or scheduled reward process with a dedicated RNG."""
        self.decay = decay
        self.initial_reward_prob = float(initial_reward_prob)
        self.reward_probs = None if reward_probs is None else list(reward_probs)
        self.counter = 1
        self.rng = rng

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset the depletion counter when leaving a patch."""
        self.counter = 1
        if rng is not None:
            self.rng = rng

    def sample_reward(self) -> float:
        """Sample reward from the configured Bernoulli reward process."""
        if self.reward_probs is None:
            assert self.decay is not None
            reward_prob = self.initial_reward_prob * np.exp(-self.decay * self.counter)
        else:
            reward_prob = self.reward_probs[
                min(self.counter, len(self.reward_probs) - 1)
            ]
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
        observable: Optional[bool] = None
    ) -> None:
        """Initialize maze dynamics from a validated spec or TOML file."""
        super().__init__()

        spec_update = {"maze": maze_spec.maze}
        if horizon is not None:
            spec_update["maze"].horizon = horizon
        if observable is not None:
            spec_update["maze"].observable = observable

        maze_spec = maze_spec.model_copy(update=spec_update)
        self._original_maze_spec = maze_spec
        self.rng = rng or np.random.default_rng(seed)

        self._update_from_maze_spec(maze_spec)

        # Internal state
        self.state = self.initial_state
        self.time = 0

    def _update_from_maze_spec(self, maze_spec: MazeSpec):
        self.maze_spec = maze_spec

        self.horizon = maze_spec.maze.horizon
        self.observable = maze_spec.maze.observable

        self.num_states = maze_spec.num_states
        self.num_actions = maze_spec.num_actions
        self.decays = maze_spec.decays
        self.state_labels = maze_spec.state_labels
        self.action_labels = list(maze_spec.maze.action_labels)
        self.initial_state = maze_spec.maze.initial_state
        self._state_specs_by_id = {
            state_spec.id: state_spec for state_spec in maze_spec.states
        }
        self._stay_action_idx = self.action_labels.index("stay")

        # Precomputed transition tables
        self._transitions_by_state_action = maze_spec.transition_map()
        self._transition_duration_by_edge: dict[tuple[int, int, int], int] = {}
        if maze_spec.uses_transition_durations:
            self._transition_duration_by_edge = {
                (row.state, row.action, row.next_state): row.duration
                for row in maze_spec.transitions
                if isinstance(row, TransitionDurationSpec)
            }

        self.reward_models = [
            ForagingReward(
                decay=state_spec.decay,
                initial_reward_prob=state_spec.initial_reward_prob,
                reward_probs=state_spec.reward_probs,
                rng=self.rng,
            )
            for state_spec in sorted(maze_spec.states, key=lambda state: state.id)
        ]

        # observability-related
        self._state_to_observation_group = self._build_state_observation_map()
        self._observation_group_to_states = self._build_observation_group_state_map()
        self.num_observations = len(set(self._state_to_observation_group.values()))
        self._planning_transition_cache: dict[
            tuple[int, int],
            list[tuple[int, float]],
        ] = {}
        # Gymnasium spaces
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Discrete(self.num_states) if self.observable else Discrete(self.num_observations)

    def _build_state_observation_map(self) -> dict[int, int]:
        """Build mapping from concrete states to observation groups."""
        return {state.id: state.observation_group for state in self.maze_spec.states}

    def _build_observation_group_state_map(self) -> dict[int, tuple[int, ...]]:
        grouped_states: dict[int, list[int]] = defaultdict(list)
        for state_idx, observation_group in self._state_to_observation_group.items():
            grouped_states[observation_group].append(state_idx)
        return {
            observation_group: tuple(sorted(state_ids))
            for observation_group, state_ids in grouped_states.items()
        }

    def _validate_observation(self, observation_idx: int) -> None:
        if observation_idx < 0 or observation_idx >= self.num_observations:
            raise ValueError(
                f"observation {observation_idx} is out of range [0, {self.num_observations})"
            )

    def _observe(self, state: Optional[int] = None) -> int:
        """Return observation group id for a state (or current state by default)."""
        state_idx = self.state if state is None else state
        self._validate_state(state_idx)
        return self._state_to_observation_group[state_idx]

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

    def valid_actions(self, state_idx: int) -> list[int]:
        """Return the globally indexed actions available from a state."""
        if self.observable:
            return self._valid_actions_observable(state_idx)
        else:
            return self._valid_actions_pomdp(state_idx)

    def _valid_actions_observable(self, state_idx: int) -> list[int]:
        """Return the globally indexed actions available from a state."""
        self._validate_state(state_idx)
        return [
            action_idx
            for action_idx in range(self.num_actions)
            if (state_idx, action_idx) in self._transitions_by_state_action
        ]

    def _valid_actions_pomdp(self, state_idx: int) -> list[int]:
        """Return the actions that are legal for an observation group."""
        self._validate_observation(state_idx)
        concrete_states = self._observation_group_to_states[state_idx]
        reference_actions = self._valid_actions_observable(concrete_states[0])
        for concrete_state in concrete_states[1:]:
            concrete_actions = self._valid_actions_observable(concrete_state)
            if concrete_actions != reference_actions:
                raise ValueError(
                    "Observation-group action availability is ill-defined because "
                    f"hidden states in observation group {state_idx} do not share "
                    "the same valid actions."
                )
        return list(reference_actions)


    def transition_distribution(
        self, state_idx: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Return sorted (next_state, probability) transitions for a state-action."""
        self._validate_state(state_idx)
        self._validate_action(action_idx)
        transition_key = (state_idx, action_idx)
        try:
            return self._transitions_by_state_action[transition_key]
        except KeyError as exc:
            raise ValueError(
                f"action {action_idx} is not valid for state {state_idx}"
            ) from exc

    def planning_transition_distribution(
        self,
        state_idx: int,
        action_idx: int,
    ) -> list[tuple[int, float]]:
        """Return belief-independent observation-group transitions for planning."""
        self._validate_observation(state_idx)
        self._validate_action(action_idx)
        cache_key = (state_idx, action_idx)
        if cache_key in self._planning_transition_cache:
            return self._planning_transition_cache[cache_key]

        representative_states = self._observation_group_to_states[state_idx]
        collapsed_distributions: list[tuple[int, list[tuple[int, float]]]] = []
        for concrete_state in representative_states:
            observation_probs: dict[int, float] = defaultdict(float)
            for next_state, prob in self.transition_distribution(
                concrete_state,
                action_idx,
            ):
                observation_probs[self._observe(next_state)] += prob
            collapsed_distributions.append(
                (
                    concrete_state,
                    sorted(observation_probs.items(), key=lambda item: item[0]),
                )
            )

        reference_state, reference_distribution = collapsed_distributions[0]
        for concrete_state, collapsed_distribution in collapsed_distributions[1:]:
            if len(collapsed_distribution) != len(reference_distribution):
                raise ValueError(
                    "Observation-group planning is ill-defined because hidden states "
                    f"{reference_state} and {concrete_state} in observation group "
                    f"{state_idx} induce different next-observation supports. "
                    "Belief-state planning would be required."
                )

            for (expected_obs, expected_prob), (actual_obs, actual_prob) in zip(
                reference_distribution,
                collapsed_distribution,
                strict=True,
            ):
                if expected_obs != actual_obs or not np.isclose(
                    expected_prob,
                    actual_prob,
                ):
                    raise ValueError(
                        "Observation-group planning is ill-defined because hidden "
                        f"states {reference_state} and {concrete_state} in "
                        f"observation group {state_idx} induce different "
                        "next-observation distributions. Belief-state planning "
                        "would be required."
                    )

        self._planning_transition_cache[cache_key] = reference_distribution
        return reference_distribution

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

    def _get_reward(self, prev_state_idx: int, action_idx: int, next_state_idx: int) -> float:
        """Return reward for a transition with explicit blocked-leave semantics."""
        if prev_state_idx != next_state_idx:
            for reward_model in self.reward_models:
                reward_model.reset()
            return 0.0

        if action_idx == self._stay_action_idx:
            return self.reward_models[prev_state_idx].sample_reward()

        return 0.0

    def expected_stay_reward(self, state_idx: int, time_spent: int) -> float:
        """Return the expected reward for one more stay in a represented state."""
        self._validate_state(state_idx)
        if time_spent < 0:
            raise ValueError(f"time_spent must be >= 0, got {time_spent}")

        state_spec = self._state_specs_by_id[state_idx]
        if state_spec.reward_probs is not None:
            reward_index = min(time_spent, len(state_spec.reward_probs) - 1)
            return float(state_spec.reward_probs[reward_index])

        assert state_spec.decay is not None
        return float(
            state_spec.initial_reward_prob * np.exp(-state_spec.decay * time_spent)
        )

    # --- Begin Gymnasium interface ---
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset to initial state and clear per-patch depletion counters."""
        super().reset(seed=seed, options=options)

        # TODO: need to consider if this is what we want,
        # but for now we are planning to do only one episode, this will only be called once per trajectory anyway.
        self._update_from_maze_spec(self._original_maze_spec)

        self.state = self.initial_state
        self.time = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for reward_model in self.reward_models:
            reward_model.reset(rng=self.rng)

        observation = self.state if self.observable else self._observe()
        info = {"true_state": self.state}
        return observation, info

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
        reward = self._get_reward(prev_state, action, next_state)
        blocked_transition = prev_state == next_state and action != self._stay_action_idx

        self.state = next_state
        self.time += duration
        truncated = self.time >= self.horizon

        info = {"prev_state": prev_state, "action": action, "true_state": self.state}

        observation = self.state if self.observable else self._observe()

        if self.maze_spec.perturbation is not None:
            if self.time == self.maze_spec.perturbation.perturbation_time:
                self._update_from_maze_spec(self.maze_spec.perturbed())

        return observation, reward, False, truncated, info

    # Backward compatible interface
    def step_transition(self, action: int) -> tuple[Transition, bool]:
        """Execute action and return ``(Transition, done)``.

        Convenience wrapper around ``step()`` for agents that expect the
        legacy ``(Transition, bool)`` interface.
        """
        obs, reward, terminated, truncated, info = self.step(action)
        transition = Transition(
            state=info["prev_state"] if self.observable else self._observe(info["prev_state"]),
            action=info["action"],
            reward=reward,
            next_state=obs,
        )

        done = terminated or truncated
        return transition, done

    # TODO: Pulled straight from MazePOMDP without any modifications, so may need some changes
    def obs_transition_distribution(
        self, obs_group: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Return (next_obs_group, probability) marginalised over states in obs_group."""
        states_in_group = [
            s for s, g in self._state_to_observation_group.items() if g == obs_group
        ]
        weight = 1.0 / len(states_in_group)
        next_obs_probs: dict[int, float] = {}
        for s in states_in_group:
            for next_state, prob in self.transition_distribution(s, action_idx):
                next_obs = self._state_to_observation_group[next_state]
                next_obs_probs[next_obs] = (
                    next_obs_probs.get(next_obs, 0.0) + prob * weight
                )
        return sorted(next_obs_probs.items())


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
            observation_labels=["Upper Patch", "Lower Patch"],
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


def maze_from_builtin_maze_spec(
    name: str = "simple",
    observable: bool = True,
    horizon: int | None = None,
) -> Maze:
    return Maze(load_builtin_maze_spec(name), horizon=horizon, observable=observable)
