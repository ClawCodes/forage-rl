"""Agent modules implementing different reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model_based import MBRL
from .q_learning import QLearning, QLearningTime
from .q_table import QTable
from .registry import EvaluatorSpec, PolicySpec, get_agent, registered_agents
from .value_iteration import ValueIterationSolver

if TYPE_CHECKING:
    from .dqn import DQNAgent
    from .drqn import DRQNAgent
    from .recurrent import ElmanAgent, GRUAgent, LSTMAgent

__all__ = [
    "DQNAgent",
    "DRQNAgent",
    "ElmanAgent",
    "GRUAgent",
    "LSTMAgent",
    "QLearning",
    "QLearningTime",
    "MBRL",
    "QTable",
    "ValueIterationSolver",
    "EvaluatorSpec",
    "PolicySpec",
    "get_agent",
    "registered_agents",
]


def __getattr__(name: str) -> Any:
    if name == "DQNAgent":
        from .dqn import DQNAgent

        globals()[name] = DQNAgent
        return DQNAgent

    if name == "DRQNAgent":
        from .drqn import DRQNAgent

        globals()[name] = DRQNAgent
        return DRQNAgent

    if name in {"ElmanAgent", "GRUAgent", "LSTMAgent"}:
        from .recurrent import ElmanAgent, GRUAgent, LSTMAgent

        recurrent_agents = {
            "ElmanAgent": ElmanAgent,
            "GRUAgent": GRUAgent,
            "LSTMAgent": LSTMAgent,
        }
        globals().update(recurrent_agents)
        return recurrent_agents[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
