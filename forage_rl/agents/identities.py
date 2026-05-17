"""Identity keys for saved policies and evaluator outputs."""

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

from forage_rl.agents.context import (
    DEFAULT_NEURAL_CONTEXT_MODE,
    NeuralContextMode,
    context_mode_artifact_label,
    context_mode_display_label,
    validate_context_mode,
)
from forage_rl.agents.registry import Agent, agent_display_label, is_neural_agent


class EvaluatorMode(StrEnum):
    FRESH = auto()
    PRETRAINED = auto()


def resolve_agent_context_mode(
    agent: Agent,
    context_mode: str | NeuralContextMode | None = None,
) -> NeuralContextMode | None:
    if not is_neural_agent(agent):
        return None
    if context_mode is None:
        return DEFAULT_NEURAL_CONTEXT_MODE
    return validate_context_mode(context_mode)


def _context_mode_suffix(
    agent: Agent,
    context_mode: NeuralContextMode | None,
) -> str:
    if (
        not is_neural_agent(agent)
        or context_mode is None
        or context_mode == DEFAULT_NEURAL_CONTEXT_MODE
    ):
        return ""
    return f"_{context_mode_artifact_label(context_mode)}"


@dataclass(frozen=True)
class PolicyIdentity:
    """A saved/generated policy source, normalized for lookup and labeling.

    This is not an agent config. It is the key used when loading run datasets,
    naming policy artifacts, and labeling plots. Agent alone is not enough for
    neural policies because DQN with the default context and DQN with
    prev_reward context are different saved policy sources.
    """

    agent: Agent
    context_mode: NeuralContextMode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_mode",
            resolve_agent_context_mode(self.agent, self.context_mode),
        )

    @property
    def artifact_label(self) -> str:
        return f"{self.agent.value}{_context_mode_suffix(self.agent, self.context_mode)}"

    @property
    def display_label(self) -> str:
        base = agent_display_label(self.agent)
        if is_neural_agent(self.agent) and self.context_mode is not None:
            return f"{base} ({context_mode_display_label(self.context_mode)})"
        return base


@dataclass(frozen=True)
class EvaluatorIdentity:
    """A model-inference evaluator, normalized for output file labels.

    This identifies the model that scores a trajectory dataset. It includes the
    agent, whether the evaluator starts fresh or from a pretrained checkpoint,
    and the neural context mode when the evaluator is neural.
    """

    agent: Agent
    mode: EvaluatorMode = EvaluatorMode.FRESH
    checkpoint_path: Path | None = None
    context_mode: NeuralContextMode | None = None

    def __post_init__(self) -> None:
        try:
            resolved_mode = EvaluatorMode(self.mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported evaluator mode {self.mode!r}. "
                f"Expected one of {', '.join(mode.value for mode in EvaluatorMode)}."
            ) from exc
        object.__setattr__(self, "mode", resolved_mode)
        object.__setattr__(
            self,
            "context_mode",
            resolve_agent_context_mode(self.agent, self.context_mode),
        )

    @property
    def label(self) -> str:
        return (
            f"{self.agent.value}"
            f"{_context_mode_suffix(self.agent, self.context_mode)}"
            f"_{self.mode.value}"
        )
