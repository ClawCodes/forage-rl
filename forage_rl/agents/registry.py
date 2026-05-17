"""Agent names, metadata, and constructor dispatch."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Any, Final

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.dqn import DQNAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.agents.recurrent import ElmanAgent, GRUAgent, LSTMAgent
from forage_rl.agents.sr_dyna import SRDynaAgent
from forage_rl.agents.sr_mb import SRMBAgent
from forage_rl.agents.sr_td import SRTDAgent

_NEURAL_ONLY_KWARGS: Final[frozenset[str]] = frozenset(
    {
        "batch_size",
        "burn_in",
        "checkpoint_path",
        "checkpoint_path_override",
        "context_mode",
        "device",
        "gradient_clip",
        "init_mode",
        "learning_rate",
        "recurrent_hidden_size",
        "recurrent_num_layers",
        "replay_capacity",
        "sequence_length",
        "target_update_interval",
        "warmup_steps",
    }
)


class Agent(StrEnum):
    MBRL = auto()
    Q_LEARNING = auto()
    SR_TD = auto()
    SR_MB = auto()
    SR_DYNA = auto()
    DQN = auto()
    ELMAN = auto()
    GRU = auto()
    LSTM = auto()


class _AgentKind(Enum):
    TABULAR = auto()
    NEURAL = auto()
    RECURRENT = auto()


@dataclass(frozen=True)
class _AgentRegistration:
    """Everything get_agent needs to construct one registered agent."""

    factory: Callable[..., BaseAgent]
    kind: _AgentKind
    display_label: str

    @property
    def is_neural(self) -> bool:
        return self.kind in {_AgentKind.NEURAL, _AgentKind.RECURRENT}

    @property
    def is_recurrent(self) -> bool:
        return self.kind == _AgentKind.RECURRENT


def is_neural_agent(agent: Agent) -> bool:
    """Return whether an agent is backed by PyTorch."""
    return _AGENT_REGISTRY[agent].is_neural


def agent_display_label(agent: Agent) -> str:
    """Return the display label for an agent."""
    return _AGENT_REGISTRY[agent].display_label


def neural_agents() -> list[Agent]:
    """Return neural-agent names."""
    return [agent for agent, spec in _AGENT_REGISTRY.items() if spec.is_neural]


def recurrent_agents() -> list[Agent]:
    """Return recurrent-agent names."""
    return [agent for agent, spec in _AGENT_REGISTRY.items() if spec.is_recurrent]


def _drop_neural_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in kwargs.items() if key not in _NEURAL_ONLY_KWARGS
    }


_AGENT_REGISTRY: Final[dict[Agent, _AgentRegistration]] = {
    Agent.MBRL: _AgentRegistration(
        MBRL,
        _AgentKind.TABULAR,
        "MBRL",
    ),
    Agent.Q_LEARNING: _AgentRegistration(
        QLearningTime,
        _AgentKind.TABULAR,
        "Q-Learning",
    ),
    Agent.SR_TD: _AgentRegistration(
        SRTDAgent,
        _AgentKind.TABULAR,
        "SR-TD",
    ),
    Agent.SR_MB: _AgentRegistration(
        SRMBAgent,
        _AgentKind.TABULAR,
        "SR-MB",
    ),
    Agent.SR_DYNA: _AgentRegistration(
        SRDynaAgent,
        _AgentKind.TABULAR,
        "SR-DYNA",
    ),
    Agent.DQN: _AgentRegistration(
        DQNAgent,
        _AgentKind.NEURAL,
        "DQN",
    ),
    Agent.ELMAN: _AgentRegistration(
        ElmanAgent,
        _AgentKind.RECURRENT,
        "ELMAN",
    ),
    Agent.GRU: _AgentRegistration(
        GRUAgent,
        _AgentKind.RECURRENT,
        "GRU",
    ),
    Agent.LSTM: _AgentRegistration(
        LSTMAgent,
        _AgentKind.RECURRENT,
        "LSTM",
    ),
}


def get_agent(name: Agent, maze, **kwargs) -> BaseAgent:
    registration = _AGENT_REGISTRY.get(name)
    if registration is None:
        raise ValueError(
            f"Unknown agent: {name!r}. Available agents: {list(_AGENT_REGISTRY)}."
        )
    remaining_kwargs = (
        kwargs if registration.is_neural else _drop_neural_only_kwargs(kwargs)
    )
    return registration.factory(maze, **remaining_kwargs)


def registered_agents() -> list[Agent]:
    return list(_AGENT_REGISTRY)
