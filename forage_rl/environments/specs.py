"""Schema models for TOML-defined maze variants."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError

from forage_rl.config import DefaultParams


class MazeMeta(BaseModel):
    """Top-level metadata and shared environment settings."""

    name: str = "maze"
    horizon: int = Field(default=DefaultParams.HORIZON, gt=0)
    initial_state: int = 0
    action_labels: List[str] = Field(default_factory=lambda: ["stay", "leave"])


class StateSpec(BaseModel):
    """Single state definition."""

    id: int
    label: str
    decay: float = Field(gt=0)
    observation_group: int = Field(gt=-1)


class TransitionStepSpec(BaseModel):
    """Transition row without explicit time duration."""

    model_config = ConfigDict(extra="forbid")

    state: int
    action: int
    next_state: int
    prob: float


class TransitionDurationSpec(BaseModel):
    """Transition row with explicit time duration."""

    model_config = ConfigDict(extra="forbid")

    state: int
    action: int
    next_state: int
    prob: float
    duration: int


TransitionSpec = Union[TransitionStepSpec, TransitionDurationSpec]


class MazeSpec(BaseModel):
    """Validated maze specification loaded from TOML."""

    model_config = ConfigDict(frozen=True)

    maze: MazeMeta
    states: List[StateSpec]
    transitions: List[TransitionSpec]

    @property
    def num_states(self) -> int:
        """Return number of states defined in the spec."""
        return len(self.states)

    @property
    def num_actions(self) -> int:
        """Return number of actions implied by `maze.action_labels`."""
        return len(self.maze.action_labels)

    @property
    def uses_transition_durations(self) -> bool:
        """Return whether transition rows include explicit durations."""
        if not self.transitions:
            return False
        return isinstance(self.transitions[0], TransitionDurationSpec)

    def _sorted_states(self):
        return sorted(self.states, key=lambda s: s.id)

    @property
    def decays(self) -> List[float]:
        """Return per-state decay values ordered by state id."""
        return [state_spec.decay for state_spec in self._sorted_states()]

    @property
    def state_labels(self) -> List[str]:
        """Return state labels ordered by state id."""
        return [state_spec.label for state_spec in self._sorted_states()]

    def transition_map(self) -> Dict[Tuple[int, int], List[Tuple[int, float]]]:
        """
        Build a sorted transition map keyed by (state, action).

        Output structure:
            {(<state_t>, <action_t>): [(<state_t+1>, <action_t+1>), ..., (<state_t+n>, <action_t+n>)]}
        """
        mapping: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
        for transition_row in self.transitions:
            mapping[(transition_row.state, transition_row.action)].append(
                (transition_row.next_state, transition_row.prob)
            )

        return {
            key: sorted(values, key=lambda item: item[0])
            for key, values in mapping.items()
        }

    @model_validator(mode="before")
    @classmethod
    def _expand_compact_format(cls, data: Any) -> Any:
        """
        Detect and expand the compact state-centric TOML format.

        Compact format has ``states`` as a dict keyed by string state ids,
        each containing a ``transitions`` sub-dict keyed by action name.
        This pre-validator rewrites it into the flat internal form expected
        by the after-validator.
        """
        if not isinstance(data, dict):
            return data

        states_raw = data.get("states")
        if not isinstance(states_raw, dict):
            return data

        # Accept 'actions' as alias for 'action_labels' in maze meta
        maze_raw = data.get("maze", {})
        if (
            isinstance(maze_raw, dict)
            and "actions" in maze_raw
            and "action_labels" not in maze_raw
        ):
            maze_raw["action_labels"] = maze_raw.pop("actions")
            data["maze"] = maze_raw

        action_labels = maze_raw.get("action_labels")

        if action_labels is None:
            raise ValidationError("Please specify action labels")

        action_index = {name: idx for idx, name in enumerate(action_labels)}

        flat_states: List[Dict[str, Any]] = []
        flat_transitions: List[Dict[str, Any]] = []

        for state_key, state_body in states_raw.items():
            state_id = int(state_key)
            transitions_raw = state_body.pop("transitions", {})

            flat_states.append({"id": state_id, **state_body})

            for action_name, outcomes in transitions_raw.items():
                action_idx = action_index.get(action_name)
                if action_idx is None:
                    raise ValueError(
                        f"Unknown action '{action_name}' in state {state_id}; "
                        f"valid actions: {list(action_index)}"
                    )
                for outcome in outcomes:
                    if len(outcome) == 2:
                        next_state, prob = outcome
                        flat_transitions.append(
                            {
                                "state": state_id,
                                "action": action_idx,
                                "next_state": next_state,
                                "prob": prob,
                            }
                        )
                    elif len(outcome) == 3:
                        next_state, prob, duration = outcome
                        flat_transitions.append(
                            {
                                "state": state_id,
                                "action": action_idx,
                                "next_state": next_state,
                                "prob": prob,
                                "duration": duration,
                            }
                        )
                    else:
                        raise ValueError(
                            f"Transition outcome must be [next_state, prob] or "
                            f"[next_state, prob, duration], got {outcome}"
                        )

        data["states"] = flat_states
        data["transitions"] = flat_transitions
        return data

    @staticmethod
    def _is_consecutive(sequence: Sequence[int]) -> bool:
        return sorted(sequence) == list(range(len(sequence)))

    @model_validator(mode="after")
    def validate_spec(self) -> "MazeSpec":
        """Validate structural, index, probability, and duration constraints."""
        if not self.states:
            raise ValueError("states must contain at least one entry")
        if not self.transitions:
            raise ValueError("transitions must contain at least one entry")
        if not self.maze.action_labels:
            raise ValueError("maze.action_labels must contain at least one entry")

        state_ids = [state_spec.id for state_spec in self.states]

        if not self._is_consecutive(state_ids):
            raise ValueError(
                f"states ids must be contiguous 0..N-1, got {sorted(state_ids)}"
            )

        if self.maze.initial_state not in state_ids:
            raise ValueError(
                f"maze.initial_state {self.maze.initial_state} is out of bounds for {self.num_states} states"
            )

        observation_groups = sorted(
            {state_spec.observation_group for state_spec in self.states}
        )

        if not self._is_consecutive(observation_groups):
            raise ValueError(
                "observation_group values must be contiguous 0..K-1, "
                f"got {observation_groups}"
            )

        seen_triples = set()
        prob_sums_by_state_action: Dict[Tuple[int, int], float] = defaultdict(float)

        has_duration_rows = [
            isinstance(transition_row, TransitionDurationSpec)
            for transition_row in self.transitions
        ]

        if any(has_duration_rows) and not all(has_duration_rows):
            raise ValueError(
                "Mixed transition modes are invalid: either all transition rows "
                "must include duration or none may include duration."
            )

        uses_duration_mode = all(has_duration_rows)

        for transition_row in self.transitions:
            if transition_row.state < 0 or transition_row.state >= self.num_states:
                raise ValueError(
                    f"transitions state {transition_row.state} is out of bounds for {self.num_states} states"
                )
            if (
                transition_row.next_state < 0
                or transition_row.next_state >= self.num_states
            ):
                raise ValueError(
                    f"transitions next_state {transition_row.next_state} is out of bounds for {self.num_states} states"
                )
            if transition_row.action < 0 or transition_row.action >= self.num_actions:
                raise ValueError(
                    f"transitions action {transition_row.action} is out of bounds for {self.num_actions} actions"
                )
            if transition_row.prob < 0:
                raise ValueError("transitions prob must be >= 0")
            if uses_duration_mode:
                duration_row = transition_row
                if not isinstance(duration_row, TransitionDurationSpec):
                    raise ValueError("Internal transition mode mismatch")
                if duration_row.duration < 1:
                    raise ValueError("transitions duration must be >= 1")

            triple = (
                transition_row.state,
                transition_row.action,
                transition_row.next_state,
            )

            if triple in seen_triples:
                raise ValueError(
                    "duplicate transition row for "
                    f"state={transition_row.state}, "
                    f"action={transition_row.action}, "
                    f"next_state={transition_row.next_state}"
                )

            seen_triples.add(triple)

            prob_sums_by_state_action[
                (transition_row.state, transition_row.action)
            ] += transition_row.prob

        for state_idx in range(self.num_states):
            for action_idx in range(self.num_actions):
                state_action_key = (state_idx, action_idx)
                if state_action_key not in prob_sums_by_state_action:
                    raise ValueError(
                        "missing transition rows for "
                        f"state={state_idx}, action={action_idx}"
                    )
                total_prob = prob_sums_by_state_action[state_action_key]
                if not np.isclose(total_prob, 1.0):
                    raise ValueError(
                        "transition probabilities for "
                        f"state={state_idx}, action={action_idx} "
                        f"must sum to 1.0, got {total_prob:.6f}"
                    )

        return self
