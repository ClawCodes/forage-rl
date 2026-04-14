"""Type definitions for reinforcement learning transitions and run datasets."""

from collections.abc import Iterator
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class Transition(BaseModel):
    """A single state transition in a reinforcement learning environment."""

    model_config = ConfigDict(frozen=True)

    state: int
    action: int
    reward: float
    next_state: int

    def __iter__(self):
        """Iterate over field values in definition order."""
        for name in type(self).model_fields:
            yield getattr(self, name)


class TimedTransition(Transition):
    """A transition that includes the time spent in the state."""

    time_spent: int

    @classmethod
    def from_transition_time(
        cls,
        transition: Transition,
        time: int,
    ) -> "TimedTransition":
        """Create a timed transition from a base transition and elapsed time."""
        return cls(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            time_spent=time,
        )


T = TypeVar("T", bound=Transition)


class Trajectory(BaseModel, Generic[T]):
    """A single episode represented as an ordered list of transitions."""

    transitions: list[T]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "Trajectory[T]":
        """Reject empty trajectories so one trajectory always means one episode."""
        if not self.transitions:
            raise ValueError("Trajectory must contain at least one transition.")
        return self

    @classmethod
    def from_numpy(cls, arr: np.ndarray, transition_cls: type[T]) -> "Trajectory[T]":
        """Create a single-episode trajectory from a 2D numpy array."""
        fields = list(transition_cls.model_fields.keys())
        transitions = [transition_cls(**dict(zip(fields, row))) for row in arr]
        return cls(transitions=transitions)

    def to_numpy(self) -> np.ndarray:
        """Convert this trajectory to a 2D numpy array."""
        return np.array([list(t) for t in self.transitions])

    def __iter__(self):
        """Iterate over transitions in this single episode."""
        return iter(self.transitions)

    def __len__(self):
        """Return the number of transitions in this episode."""
        return len(self.transitions)

    def transition_cls(self) -> type[T]:
        """Return the transition class carried by this episode."""
        return type(self.transitions[0])


class RunDataset(BaseModel, Generic[T]):
    """A training run represented as an ordered list of episode trajectories."""

    trajectories: list[Trajectory[T]]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "RunDataset[T]":
        """Reject empty run datasets and mixed transition classes."""
        if not self.trajectories:
            raise ValueError("RunDataset must contain at least one trajectory.")

        first_cls = self.trajectories[0].transition_cls()
        for trajectory in self.trajectories[1:]:
            if trajectory.transition_cls() is not first_cls:
                raise ValueError(
                    "RunDataset trajectories must share one transition type."
                )
        return self

    def __iter__(self) -> Iterator[Trajectory[T]]:
        """Iterate over episode trajectories in this run."""
        return iter(self.trajectories)

    def __len__(self) -> int:
        """Return the number of episodes in this run."""
        return len(self.trajectories)

    def num_episodes(self) -> int:
        """Return the number of episodes in this run."""
        return len(self.trajectories)

    def num_transitions(self) -> int:
        """Return the total number of transitions across all episodes."""
        return sum(len(trajectory) for trajectory in self.trajectories)

    def iter_transitions(self) -> Iterator[T]:
        """Yield transitions across all episodes in order."""
        for trajectory in self.trajectories:
            yield from trajectory

    def transition_cls(self) -> type[T]:
        """Return the transition class used by all episodes in this run."""
        return self.trajectories[0].transition_cls()
