from typing import Annotated, Any, ClassVar, List

import numpy as np
from numpy import signedinteger
from pydantic import BaseModel, BeforeValidator, ConfigDict

# An integer type accepting python and numpy integer types
type SignedInteger = int | signedinteger[Any]

# Pydantic-compatible integer that coerces numpy integers to Python int
Int = Annotated[SignedInteger, BeforeValidator(lambda x: int(x))]


class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: Int
    action: Int
    reward: float
    next_state: Int

    def __iter__(self):
        for name, _ in self.model_fields.items():
            yield getattr(self, name)


class TimedTransition(Transition):
    time_spent: Int

    def __iter__(self):
        for name, _ in self.model_fields.items():
            yield getattr(self, name)

    @classmethod
    def from_transition_time(
        cls, transition: Transition, time: SignedInteger
    ) -> "TimedTransition":
        return cls(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            time_spent=time,
        )


class Trajectory(BaseModel):
    fields: ClassVar[List[str]] = [
        "state",
        "action",
        "reward",
        "next_state",
        "time_spent",
    ]
    transitions: list[TimedTransition]

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Trajectory":
        transitions = [TimedTransition(**dict(zip(cls.fields, row))) for row in arr]
        return cls(transitions=transitions)

    def to_numpy(self) -> np.ndarray:
        return np.array([list(t) for t in self.transitions])

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)
