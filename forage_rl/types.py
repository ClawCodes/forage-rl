from typing import Annotated, Any, ClassVar, List

import numpy as np
from numpy import signedinteger
from pydantic import BaseModel, BeforeValidator

# An integer type accepting python and numpy integer types
type SignedInteger = int | signedinteger[Any]

# Pydantic-compatible integer that coerces numpy integers to Python int
Int = Annotated[int, BeforeValidator(lambda x: int(x))]


class Transition(BaseModel):
    state: Int
    time_spent: Int
    action: Int
    reward: float
    next_state: Int

    def __iter__(self):
        for name, _ in self.model_fields.items():
            yield getattr(self, name)


class Trajectory(BaseModel):
    fields: ClassVar[List[str]] = [
        "state",
        "time_spent",
        "action",
        "reward",
        "next_state",
    ]
    transitions: list[Transition]

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Trajectory":
        transitions = [Transition(**dict(zip(cls.fields, row))) for row in arr]
        return cls(transitions=transitions)

    def to_numpy(self) -> np.ndarray:
        return np.array([list(t) for t in self.transitions])

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)
