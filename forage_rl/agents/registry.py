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
    ELMAN = "elman"
    GRU = "gru"
    LSTM = "lstm"
    DRQN = "drqn"


NeuralContextMode = Literal[
    "observation_only",
    "prev_reward",
    "prev_reward_time",
    "legacy_context",
]
NEURAL_CONTEXT_MODES: tuple[NeuralContextMode, ...] = (
    "observation_only",
    "prev_reward",
    "prev_reward_time",
    "legacy_context",
)
NEURAL_CONTEXT_MODE_TOKENS: dict[NeuralContextMode, str] = {
    "observation_only": "obs_only",
    "prev_reward": "prev_reward",
    "prev_reward_time": "prev_reward_time",
    "legacy_context": "legacy_context",
}
NEURAL_CONTEXT_MODE_DISPLAY_LABELS: dict[NeuralContextMode, str] = {
    "observation_only": "obs-only",
    "prev_reward": "obs+prev_reward",
    "prev_reward_time": "obs+prev_reward+time",
    "legacy_context": "legacy-context",
}
NEURAL_CONTEXT_MODE_ORDER: dict[NeuralContextMode, int] = {
    context_mode: index for index, context_mode in enumerate(NEURAL_CONTEXT_MODES)
}
CANONICAL_AGENT_ALIASES: dict[Agent, Agent] = {
    Agent.DRQN: Agent.LSTM,
}
CANONICAL_AGENT_ORDER: tuple[Agent, ...] = (
    Agent.MBRL,
    Agent.QLearning,
    Agent.DQN,
    Agent.ELMAN,
    Agent.GRU,
    Agent.LSTM,
)
CANONICAL_NEURAL_AGENTS: tuple[Agent, ...] = (
    Agent.DQN,
    Agent.ELMAN,
    Agent.GRU,
    Agent.LSTM,
)
CANONICAL_RECURRENT_AGENTS: tuple[Agent, ...] = (
    Agent.ELMAN,
    Agent.GRU,
    Agent.LSTM,
)


def canonical_agent(agent: Agent) -> Agent:
    """Map compatibility aliases onto their canonical agent names."""
    return CANONICAL_AGENT_ALIASES.get(agent, agent)


def legacy_alias_agents(agent: Agent) -> tuple[Agent, ...]:
    """Return legacy aliases that may exist on disk for a canonical agent."""
    if canonical_agent(agent) == Agent.LSTM:
        return (Agent.DRQN,)
    return ()


def is_neural_agent(agent: Agent) -> bool:
    """Return whether an agent is backed by PyTorch."""
    return canonical_agent(agent) in CANONICAL_NEURAL_AGENTS


def is_recurrent_agent(agent: Agent) -> bool:
    """Return whether an agent uses a recurrent Q-network."""
    return canonical_agent(agent) in CANONICAL_RECURRENT_AGENTS


def agent_display_label(agent: Agent) -> str:
    """Return the canonical display label for an agent."""
    resolved = canonical_agent(agent)
    if resolved == Agent.QLearning:
        return "Q-Learning"
    return resolved.value.upper()


def neural_agents() -> list[Agent]:
    """Return canonical neural-agent names without legacy aliases."""
    return list(CANONICAL_NEURAL_AGENTS)


def recurrent_agents() -> list[Agent]:
    """Return canonical recurrent-agent names without legacy aliases."""
    return list(CANONICAL_RECURRENT_AGENTS)


def validate_context_mode(context_mode: str) -> NeuralContextMode:
    if context_mode not in NEURAL_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported context_mode {context_mode!r}. Expected one of "
            f"{', '.join(NEURAL_CONTEXT_MODES)}."
        )
    return context_mode


def context_mode_token(context_mode: NeuralContextMode) -> str:
    return NEURAL_CONTEXT_MODE_TOKENS[context_mode]


def context_mode_display_label(context_mode: NeuralContextMode) -> str:
    return NEURAL_CONTEXT_MODE_DISPLAY_LABELS[context_mode]


def context_mode_sort_key(context_mode: NeuralContextMode) -> int:
    return NEURAL_CONTEXT_MODE_ORDER[context_mode]


@dataclass(frozen=True)
class PolicySpec:
    agent: Agent
    context_mode: NeuralContextMode = "legacy_context"

    @property
    def artifact_label(self) -> str:
        resolved = canonical_agent(self.agent)
        if is_neural_agent(resolved) and self.context_mode != "legacy_context":
            return f"{resolved.value}_{context_mode_token(self.context_mode)}"
        return resolved.value

    @property
    def display_label(self) -> str:
        base = agent_display_label(self.agent)
        if is_neural_agent(self.agent):
            return f"{base} ({context_mode_display_label(self.context_mode)})"
        return base


@dataclass(frozen=True)
class EvaluatorSpec:
    agent: Agent
    mode: Literal["fresh", "pretrained"] = "fresh"
    checkpoint_path: Path | None = None
    context_mode: NeuralContextMode = "legacy_context"

    @property
    def label(self) -> str:
        resolved = canonical_agent(self.agent)
        if is_neural_agent(resolved) and self.context_mode != "legacy_context":
            return (
                f"{resolved.value}_{context_mode_token(self.context_mode)}_{self.mode}"
            )
        return f"{resolved.value}_{self.mode}"


def _build_dqn_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.dqn import DQNAgent

    return DQNAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        **kwargs,
    )


def _recurrent_defaults(maze) -> tuple[int, int]:
    return (
        DefaultParams.RECURRENT_SEQUENCE_LENGTH,
        DefaultParams.RECURRENT_BURN_IN,
    )


def _build_elman_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.recurrent import ElmanAgent

    default_sequence_length, default_burn_in = _recurrent_defaults(maze)

    return ElmanAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        sequence_length=kwargs.pop("sequence_length", default_sequence_length),
        burn_in=kwargs.pop("burn_in", default_burn_in),
        recurrent_hidden_size=kwargs.pop(
            "recurrent_hidden_size",
            DefaultParams.RECURRENT_HIDDEN_SIZE,
        ),
        recurrent_num_layers=kwargs.pop(
            "recurrent_num_layers",
            DefaultParams.RECURRENT_NUM_LAYERS,
        ),
        **kwargs,
    )


def _build_gru_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.recurrent import GRUAgent

    default_sequence_length, default_burn_in = _recurrent_defaults(maze)

    return GRUAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        sequence_length=kwargs.pop("sequence_length", default_sequence_length),
        burn_in=kwargs.pop("burn_in", default_burn_in),
        recurrent_hidden_size=kwargs.pop(
            "recurrent_hidden_size",
            DefaultParams.RECURRENT_HIDDEN_SIZE,
        ),
        recurrent_num_layers=kwargs.pop(
            "recurrent_num_layers",
            DefaultParams.RECURRENT_NUM_LAYERS,
        ),
        **kwargs,
    )


def _build_lstm_agent(maze, **kwargs) -> BaseAgent:
    from forage_rl.agents.recurrent import LSTMAgent

    default_sequence_length, default_burn_in = _recurrent_defaults(maze)

    return LSTMAgent(
        maze,
        num_episodes=kwargs.pop("num_episodes", DefaultParams.NUM_EPISODES),
        gamma=kwargs.pop("gamma", DefaultParams.GAMMA),
        beta=kwargs.pop("beta", DefaultParams.BETA),
        sequence_length=kwargs.pop("sequence_length", default_sequence_length),
        burn_in=kwargs.pop("burn_in", default_burn_in),
        recurrent_hidden_size=kwargs.pop(
            "recurrent_hidden_size",
            DefaultParams.RECURRENT_HIDDEN_SIZE,
        ),
        recurrent_num_layers=kwargs.pop(
            "recurrent_num_layers",
            DefaultParams.RECURRENT_NUM_LAYERS,
        ),
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
            if key not in {"device", "init_mode", "checkpoint_path", "context_mode"}
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
            if key not in {"device", "init_mode", "checkpoint_path", "context_mode"}
        },
    )


AGENT_REGISTRY: dict[Agent, AgentFactory] = {
    Agent.MBRL: _build_mbrl_agent,
    Agent.QLearning: _build_q_learning_agent,
    Agent.DQN: _build_dqn_agent,
    Agent.ELMAN: _build_elman_agent,
    Agent.GRU: _build_gru_agent,
    Agent.LSTM: _build_lstm_agent,
}


def get_agent(name: Agent, maze, **kwargs) -> BaseAgent:
    resolved = canonical_agent(name)
    if resolved not in AGENT_REGISTRY:
        raise ValueError(
            "Unknown agent: "
            f"{name!r}. Available canonical agents: {list(CANONICAL_AGENT_ORDER)}. "
            "Legacy alias: 'drqn' -> 'lstm'."
        )
    return AGENT_REGISTRY[resolved](maze, **kwargs)


def registered_agents() -> list[Agent]:
    return list(CANONICAL_AGENT_ORDER)
