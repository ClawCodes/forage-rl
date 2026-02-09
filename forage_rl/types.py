"""Type definitions for reinforcement learning transitions and trajectories.

This module defines the core data structures used to represent state transitions
and trajectories in reinforcement learning environments.
"""

from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict


class Transition(BaseModel):
    """A single state transition in a reinforcement learning environment.

    Represents the standard (s, a, r, s') tuple from RL theory, capturing one
    step of interaction between an agent and environment.

    Attributes:
        state: The state the agent was in before taking the action.
        action: The action taken by the agent.
        reward: The reward received after taking the action.
        next_state: The resulting state after the action was taken.
    """

    model_config = ConfigDict(frozen=True)

    state: int
    action: int
    reward: float
    next_state: int

    def __iter__(self):
        """Iterate over field values in definition order."""
        for name in self.__class__.model_fields:
            yield getattr(self, name)


class TimedTransition(Transition):
    """A transition that includes the time spent in the state.

    Extends Transition with temporal information, useful for environments
    where actions have variable durations (e.g., foraging tasks where
    different actions take different amounts of time).

    Attributes:
        time_spent: The number of time steps spent on this transition.
    """

    time_spent: int

    @classmethod
    def from_transition_time(
        cls, transition: Transition, time: int
    ) -> "TimedTransition":
        """Create a TimedTransition from a Transition and time value.

        Args:
            transition: The base transition to extend.
            time: The time spent on this transition.

        Returns:
            A new TimedTransition with all fields from the original plus time_spent.
        """
        return cls(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            time_spent=time,
        )


T = TypeVar("T", bound=Transition)


class Trajectory(BaseModel, Generic[T]):
    """A sequence of transitions forming a complete episode or path.

    Generic over the transition type, allowing use with Transition,
    TimedTransition, or custom transition subclasses.

    Provides utilities for converting between object representation and numpy
    arrays, which is useful for batch processing and storage.

    Attributes:
        transitions: The ordered list of transitions in this trajectory.
    """

    transitions: list[T]

    @classmethod
    def from_numpy(cls, arr: np.ndarray, transition_cls: type[T]) -> "Trajectory[T]":
        """Create a Trajectory from a numpy array.

        Args:
            arr: A 2D array where each row is a transition and columns
                correspond to the fields in order.
            transition_cls: The transition class to instantiate for each row.

        Returns:
            A Trajectory containing the transitions from the array.
        """
        fields = list(transition_cls.model_fields.keys())
        transitions = [transition_cls(**dict(zip(fields, row))) for row in arr]
        return cls(transitions=transitions)

    def to_numpy(self) -> np.ndarray:
        """Convert this trajectory to a numpy array.

        Returns:
            A 2D array where each row is a transition and columns correspond
            to the fields in order.
        """
        return np.array([list(t) for t in self.transitions])

    def __iter__(self):
        """Iterate over the transitions in this trajectory."""
        return iter(self.transitions)

    def __len__(self):
        """Return the number of transitions in this trajectory."""
        return len(self.transitions)
