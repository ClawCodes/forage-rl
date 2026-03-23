"""Shared registry mapping agent names to constructor factories."""

from enum import StrEnum
from typing import Callable

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.config import DefaultParams

AgentFactory = Callable[..., BaseAgent]


class Agent(StrEnum):
    MBRL = "mbrl"
    QLearning = "q_learning"


AGENT_REGISTRY: dict[Agent, AgentFactory] = {
    Agent.MBRL: lambda maze, **kwargs: MBRL(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        **kwargs,
    ),
    Agent.QLearning: lambda maze, **kwargs: QLearningTime(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        alpha=kwargs.pop("alpha", DefaultParams.ALPHA),
        **kwargs,
    ),
}


def get_agent(name: Agent, maze, **kwargs) -> BaseAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name](maze, **kwargs)


def registered_agents() -> list[Agent]:
    return list(AGENT_REGISTRY.keys())
