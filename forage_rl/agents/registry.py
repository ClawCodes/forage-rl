"""Shared registry mapping agent names to constructor factories."""

import inspect

from forage_rl.agents.base import BaseAgent
from forage_rl.agents.dqn import DQNAgent
from forage_rl.agents.model_based import MBRL
from forage_rl.agents.q_learning import QLearningTime
from forage_rl.agents.rdqn import RDQNAgent
from forage_rl.agents.value_iteration import ValueIterationAgent

AgentFactory = type[BaseAgent]
IGNORED_SHARED_KWARGS = frozenset({"device"})

# Registry entries must accept the common kwargs forwarded by ``get_agent()``,
# including shared construction parameters like ``num_episodes`` and ``seed``.
AGENT_REGISTRY: dict[str, AgentFactory] = {
    "mbrl": MBRL,
    "q_learning": QLearningTime,
    "dqn": DQNAgent,
    "rdqn": RDQNAgent,
    "value_iteration": ValueIterationAgent,
}


def _split_constructor_kwargs(
    factory: AgentFactory,
    kwargs: dict,
) -> tuple[dict, list[str]]:
    """Return forwarded kwargs plus unknown names that should fail fast."""
    signature = inspect.signature(factory.__init__)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs), []

    accepted_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "maze"}
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    forwarded_kwargs = {
        name: value for name, value in kwargs.items() if name in accepted_kwargs
    }
    ignored_shared_kwargs = {
        name
        for name in kwargs
        if name in IGNORED_SHARED_KWARGS and name not in accepted_kwargs
    }
    unknown_kwargs = sorted(
        name
        for name in kwargs
        if name not in forwarded_kwargs and name not in ignored_shared_kwargs
    )
    return forwarded_kwargs, unknown_kwargs


def get_agent(name: str, maze, **kwargs) -> BaseAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    factory = AGENT_REGISTRY[name]
    filtered_kwargs, unknown_kwargs = _split_constructor_kwargs(factory, kwargs)
    if unknown_kwargs:
        raise TypeError(
            f"Unknown constructor kwargs for agent {name!r}: {unknown_kwargs}"
        )
    return factory(maze, **filtered_kwargs)


def registered_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())
