"""Foraging maze environments driven by validated maze specifications."""

from pathlib import Path
from dataclasses import dataclass
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

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset the depletion counter when leaving a patch."""
        self.counter = 0
        if rng is not None:
            self.rng = rng

    def sample_reward(self) -> float:
        """Sample reward as Bernoulli(exp(-decay * counter))."""
        reward_prob = np.exp(-self.decay * self.counter)
        self.counter += 1
        return 1.0 if self.rng.random() < reward_prob else 0.0


@dataclass(frozen=True)
class TransitionDetails:
    """Transition plus visible/hidden state metadata."""

    transition: Transition
    done: bool
    true_state: int
    true_next_state: int


@dataclass(frozen=True)
class TransitionTiming:
    """Elapsed time and reset semantics for a transition."""

    elapsed_time: int
    resets_time_spent: bool


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
        self.horizon = self.maze_spec.maze.horizon if horizon is None else horizon
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")

        self.rng = rng or np.random.default_rng(seed)
        self.num_states = self.maze_spec.num_states
        self.num_actions = self.maze_spec.num_actions
        self.agent_num_states = self.num_states
        self.decays = self.maze_spec.decays
        self.state_labels = self.maze_spec.state_labels
        self.agent_state_labels = list(self.state_labels)
        self.true_num_states = self.num_states
        self.true_state_labels = list(self.state_labels)
        self.action_labels = list(self.maze_spec.maze.action_labels)
        self.initial_state = self.maze_spec.maze.initial_state
        self.agent_initial_state = self.initial_state

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

    def _transition_timing(
        self,
        *,
        state: int,
        next_state: int,
        action: int | None = None,
        true_state: int | None = None,
        true_next_state: int | None = None,
    ) -> TransitionTiming:
        """Return elapsed time and reset semantics for a transition."""
        concrete_state = state if true_state is None else true_state
        concrete_next_state = next_state if true_next_state is None else true_next_state

        if action is None:
            if self.maze_spec.uses_transition_durations:
                raise ValueError(
                    "action is required to compute time progression for duration-aware mazes"
                )
            elapsed_time = 1
        else:
            elapsed_time = self.transition_duration(
                concrete_state, action, concrete_next_state
            )

        return TransitionTiming(
            elapsed_time=elapsed_time,
            resets_time_spent=concrete_state != concrete_next_state,
        )

    def _next_time_spent_from_timing(
        self, *, time_spent: int, timing: TransitionTiming
    ) -> int:
        """Convert transition timing into the next time-spent feature."""
        if timing.resets_time_spent:
            return 0
        return min(time_spent + timing.elapsed_time, self.horizon - 1)

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
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for reward_model in self.reward_models:
            reward_model.reset(rng=self.rng)
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
        timing = Maze._transition_timing(
            self,
            state=prev_state,
            next_state=next_state,
            action=action,
        )
        reward = self._get_reward(next_state)

        self.state = next_state
        self.time += timing.elapsed_time
        truncated = self.time >= self.horizon

        info = {"prev_state": prev_state, "action": action}
        return self.state, reward, False, truncated, info

    def step_transition_details(self, action: int) -> TransitionDetails:
        """Execute action and return visible and hidden-state transition metadata."""
        obs, reward, terminated, truncated, info = self.step(action)
        transition = Transition(
            state=info["prev_state"],
            action=info["action"],
            reward=reward,
            next_state=obs,
        )
        done = terminated or truncated
        return TransitionDetails(
            transition=transition,
            done=done,
            true_state=info["prev_state"],
            true_next_state=self.state,
        )

    # Backward compatible interface
    def step_transition(self, action: int) -> tuple[Transition, bool]:
        """Execute action and return ``(Transition, done)``.

        Convenience wrapper around ``step()`` for agents that expect the
        legacy ``(Transition, bool)`` interface.
        """
        details = self.step_transition_details(action)
        return details.transition, details.done

    def next_time_spent(
        self,
        *,
        state: int,
        next_state: int,
        time_spent: int,
        action: int | None = None,
        true_state: int | None = None,
        true_next_state: int | None = None,
    ) -> int:
        """Return the next ``time_spent`` value for time-aware agents."""
        timing = self._transition_timing(
            state=state,
            next_state=next_state,
            action=action,
            true_state=true_state,
            true_next_state=true_next_state,
        )
        return self._next_time_spent_from_timing(time_spent=time_spent, timing=timing)


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
        self._observation_transition_timings = (
            self._build_observation_transition_timings()
        )
        self.num_observations = len(set(self._state_to_observation_group.values()))
        self.agent_num_states = self.num_observations
        self.agent_state_labels = [
            f"Observation {i}" for i in range(self.num_observations)
        ]
        self.agent_initial_state = self.observe(self.initial_state)
        self.observation_space = Discrete(self.num_observations)

    def _build_state_observation_map(self) -> dict[int, int]:
        """Build mapping from concrete states to observation groups."""

        return {state.id: state.observation_group for state in self.maze_spec.states}

    def _build_observation_transition_timings(
        self,
    ) -> dict[tuple[int, int, int], TransitionTiming]:
        """Infer observation-level timing semantics from concrete transitions."""
        grouped_timings: dict[tuple[int, int, int], list[TransitionTiming]] = {}
        for transition_row in self.maze_spec.transitions:
            observation_transition = (
                self._state_to_observation_group[transition_row.state],
                transition_row.action,
                self._state_to_observation_group[transition_row.next_state],
            )
            grouped_timings.setdefault(observation_transition, []).append(
                TransitionTiming(
                    elapsed_time=self.transition_duration(
                        transition_row.state,
                        transition_row.action,
                        transition_row.next_state,
                    ),
                    resets_time_spent=transition_row.state != transition_row.next_state,
                )
            )

        observation_timings: dict[tuple[int, int, int], TransitionTiming] = {}
        for observation_transition, timings in grouped_timings.items():
            elapsed_times = {timing.elapsed_time for timing in timings}
            reset_semantics = {timing.resets_time_spent for timing in timings}
            if len(elapsed_times) != 1 or len(reset_semantics) != 1:
                # Observation-only planning is ill-defined when aliased hidden states
                # disagree on elapsed time or whether time_spent should reset.
                raise ValueError(
                    "Observation-level timing is ambiguous for "
                    f"observation={observation_transition[0]}, "
                    f"action={observation_transition[1]}, "
                    f"next_observation={observation_transition[2]}"
                )
            observation_timings[observation_transition] = timings[0]

        return observation_timings

    def _transition_timing(
        self,
        *,
        state: int,
        next_state: int,
        action: int | None = None,
        true_state: int | None = None,
        true_next_state: int | None = None,
    ) -> TransitionTiming:
        """Return timing semantics in either hidden or observation space."""
        if true_state is not None and true_next_state is not None:
            return super()._transition_timing(
                state=state,
                next_state=next_state,
                action=action,
                true_state=true_state,
                true_next_state=true_next_state,
            )
        if action is None:
            raise ValueError(
                "action is required to compute observation-level time progression"
            )

        transition_key = (state, action, next_state)
        try:
            return self._observation_transition_timings[transition_key]
        except KeyError as exc:
            raise ValueError(
                "No observation-level timing semantics defined for "
                f"observation={state}, action={action}, next_observation={next_state}"
            ) from exc

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
        """Reset and return observation group instead of true state."""
        _, info = super().reset(seed=seed, options=options)
        info["true_state"] = self.state
        return self.observe(), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Step and return observation group instead of true state."""
        _, reward, terminated, truncated, info = super().step(action)
        info["true_state"] = self.state
        return self.observe(), reward, terminated, truncated, info

    def step_transition_details(self, action: int) -> TransitionDetails:
        """Execute action and return visible transition plus hidden-state metadata."""
        true_prev_state = self.state
        prev_observation = self.observe(true_prev_state)
        obs, reward, terminated, truncated, info = self.step(action)
        done = terminated or truncated
        transition = Transition(
            state=prev_observation,
            action=info["action"],
            reward=reward,
            next_state=obs,
        )
        return TransitionDetails(
            transition=transition,
            done=done,
            true_state=true_prev_state,
            true_next_state=info["true_state"],
        )


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
