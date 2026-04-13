"""Shared registry mapping agent names to constructor factories."""

from enum import StrEnum
from typing import Callable

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.agents.sr_td import SRTDAgent
from forage_rl.agents.sr_mb import SRMBAgent
from forage_rl.agents.sr_dyna import SRDynaAgent
from forage_rl.config import DefaultParams

AgentFactory = Callable[..., BaseAgent]


class Agent(StrEnum):
    MBRL = "mbrl"
    QLearning = "q_learning"
    SRTD = "sr_td"
    SRMB = "sr_mb"
    SRDyna = "sr_dyna"


AGENT_REGISTRY: dict[Agent, AgentFactory] = {
    Agent.MBRL: lambda maze, num_episodes=DefaultParams.NUM_EPISODES: MBRL(
        maze, num_episodes=num_episodes, gamma=DefaultParams.GAMMA
    ),
    Agent.QLearning: lambda maze,
    num_episodes=DefaultParams.NUM_EPISODES: QLearningTime(
        maze, num_episodes=num_episodes, alpha=DefaultParams.ALPHA
    ),
    Agent.SRTD: lambda maze, num_episodes=DefaultParams.NUM_EPISODES: SRTDAgent(
        maze, num_episodes=num_episodes
    ),
    Agent.SRMB: lambda maze, num_episodes=DefaultParams.NUM_EPISODES: SRMBAgent(
        maze, num_episodes=num_episodes
    ),
    Agent.SRDyna: lambda maze, num_episodes=DefaultParams.NUM_EPISODES: SRDynaAgent(
        maze, num_episodes=num_episodes
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
