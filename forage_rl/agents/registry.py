"""Shared registry mapping agent names to constructor factories."""

from typing import Callable

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.config import DefaultParams

AgentFactory = Callable[..., BaseAgent]

AGENT_REGISTRY: dict[str, AgentFactory] = {
    "mbrl": lambda maze, num_episodes=DefaultParams.NUM_EPISODES: MBRL(
        maze, num_episodes=num_episodes, gamma=DefaultParams.GAMMA
    ),
    "q_learning": lambda maze, num_episodes=DefaultParams.NUM_EPISODES: QLearningTime(
        maze, num_episodes=num_episodes, alpha=DefaultParams.ALPHA
    ),
}


def get_agent(name: str, maze, **kwargs) -> BaseAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name](maze, **kwargs)


def registered_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())
