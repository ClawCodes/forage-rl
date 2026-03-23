"""Shared registry mapping agent names to constructor factories."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.config import DefaultParams

AgentFactory = Callable[..., BaseAgent]


class Agent(StrEnum):
    MBRL = "mbrl"
    QLearning = "q_learning"
    DQN = "dqn"
    DRQN = "drqn"


@dataclass(frozen=True)
class EvaluatorSpec:
    agent: Agent
    mode: Literal["fresh", "pretrained"] = "fresh"
    checkpoint_path: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.agent.value}_{self.mode}"


def _build_dqn_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.dqn import DQNAgent

    return DQNAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        alpha=kwargs.pop("alpha", DefaultParams.ALPHA),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        **kwargs,
    )


def _build_drqn_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.drqn import DRQNAgent

    return DRQNAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        alpha=kwargs.pop("alpha", DefaultParams.ALPHA),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        **kwargs,
    )


def _build_mbrl_agent(maze, **kwargs) -> BaseAgent:
    return MBRL(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"device", "init_mode", "checkpoint_path"}
        },
    )


def _build_q_learning_agent(maze, **kwargs) -> BaseAgent:
    return QLearningTime(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        alpha=kwargs.pop("alpha", DefaultParams.ALPHA),
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"device", "init_mode", "checkpoint_path"}
        },
    )


AGENT_REGISTRY: dict[Agent, AgentFactory] = {
    Agent.MBRL: _build_mbrl_agent,
    Agent.QLearning: _build_q_learning_agent,
    Agent.DQN: _build_dqn_agent,
    Agent.DRQN: _build_drqn_agent,
}


def get_agent(name: Agent, maze, **kwargs) -> BaseAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name](maze, **kwargs)


def registered_agents() -> list[Agent]:
    return list(AGENT_REGISTRY.keys())
