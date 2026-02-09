"""Foraging maze environments driven by validated maze specifications."""

from pathlib import Path
from typing import Optional

import numpy as np

from forage_rl.types import Transition

from .spec_loader import load_builtin_maze_spec, load_maze_spec
from .specs import (
    MazeMeta,
    MazeSpec,
    StateSpec,
    TransitionDurationSpec,
    TransitionStepSpec,
)


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


class Maze:
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

    def reset(self) -> int:
        """Reset to initial state and clear per-patch depletion counters."""
        self.state = self.initial_state
        self.time = 0
        for reward_model in self.reward_models:
            reward_model.reset()
        return self.state

    def transition_distribution(
        self, state_idx: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Return sorted (next_state, probability) transitions for a state-action."""
        self._validate_state(state_idx)
        self._validate_action(action_idx)
        return list(self._transitions_by_state_action[(state_idx, action_idx)])

    def transition_probs(
        self, state_idx: int, action_idx: int
    ) -> list[tuple[int, float]]:
        """Backward-compatible alias for transition_distribution."""
        return self.transition_distribution(state_idx, action_idx)

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

    def step(self, action_idx: int) -> tuple[Transition, bool]:
        """Execute action and return (Transition, done)."""
        self._validate_action(action_idx)
        prev_state_idx = self.state
        next_state_idx = self._sample_next_state(prev_state_idx, action_idx)
        transition_duration = self.transition_duration(
            prev_state_idx, action_idx, next_state_idx
        )
        reward_value = self._get_reward(next_state_idx)
        transition = Transition(
            state=prev_state_idx,
            action=action_idx,
            reward=reward_value,
            next_state=next_state_idx,
        )
        self.state = next_state_idx
        self.time += transition_duration
        done = self.time >= self.horizon
        return transition, done


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

    def step_observation(self, action: int) -> tuple[int, float, bool]:
        """Step the environment and emit `(observation, reward, done)`."""
        transition, done = self.step(action)
        return self.observe(transition.next_state), transition.reward, done


def _simple_spec_from_decays(decays: list[float], horizon: int) -> MazeSpec:
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
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        spec_path: Optional[str | Path] = None,
    ):
        """Initialize compatibility wrapper for default or custom simple specs."""
        if spec_path is not None:
            super().__init__(spec_path=spec_path, seed=seed, rng=rng, horizon=horizon)
            return

        if decays is not None:
            default_horizon = horizon if horizon is not None else 100
            super().__init__(
                spec=_simple_spec_from_decays(decays=decays, horizon=default_horizon),
                seed=seed,
                rng=rng,
                horizon=horizon,
            )
            return

        super().__init__(
            spec=load_builtin_maze_spec("simple"),
            seed=seed,
            rng=rng,
            horizon=horizon,
        )
