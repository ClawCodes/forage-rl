"""Agent modules implementing different reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model_based import MBRL
from .q_learning import QLearning, QLearningTime
from .q_table import QTable
from .registry import EvaluatorSpec, PolicySpec, get_agent, registered_agents
from .sr_dyna import SRDynaAgent
from .sr_mb import SRMBAgent
from .sr_td import SRTDAgent
from .successor_base import BaseSRAgent
from .value_iteration import ValueIterationSolver

if TYPE_CHECKING:
    from .dqn import DQNAgent
    from .recurrent import ElmanAgent, GRUAgent, LSTMAgent

    DRQNAgent = LSTMAgent

__all__ = [
    "BaseSRAgent",
    "DQNAgent",
    "DRQNAgent",
    "ElmanAgent",
    "EvaluatorSpec",
    "get_agent",
    "GRUAgent",
    "LSTMAgent",
    "MBRL",
    "PolicySpec",
    "QLearning",
    "QLearningTime",
    "QTable",
    "registered_agents",
    "SRDynaAgent",
    "SRMBAgent",
    "SRTDAgent",
    "ValueIterationSolver",
]


def __getattr__(name: str) -> Any:
    if name == "DQNAgent":
        from .dqn import DQNAgent

        globals()[name] = DQNAgent
        return DQNAgent

    if name == "DRQNAgent":
        from .recurrent import LSTMAgent

        globals()[name] = LSTMAgent
        return LSTMAgent

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
