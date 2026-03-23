"""Agent modules implementing different reinforcement learning algorithms."""

from .model_based import MBRL
from .q_learning import QLearning, QLearningTime
from .q_table import QTable
from .registry import Agent as _Agent, get_agent, registered_agents
from .value_iteration import ValueIterationSolver

__all__ = [
    "QLearning",
    "QLearningTime",
    "MBRL",
    "ValueIterationSolver",
    "get_agent",
    "registered_agents",
    "QTable",
    "DQNAgent",
    "DRQNAgent",
    "EvaluatorSpec",
]
