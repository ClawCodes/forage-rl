"""Context modes that define neural-agent input features and labels."""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final


class NeuralContextMode(StrEnum):
    OBSERVATION_ONLY = auto()
    PREV_REWARD = auto()
    PREV_REWARD_TIME = auto()
    PREV_ACTION_REWARD_TIME = auto()


@dataclass(frozen=True)
class _NeuralContextModeSpec:
    display_label: str
    feature_components: tuple[str, ...]


_NEURAL_CONTEXT_MODE_SPECS: Final[dict[NeuralContextMode, _NeuralContextModeSpec]] = {
    NeuralContextMode.OBSERVATION_ONLY: _NeuralContextModeSpec(
        display_label="obs",
        feature_components=("observation_one_hot",),
    ),
    NeuralContextMode.PREV_REWARD: _NeuralContextModeSpec(
        display_label="obs+prev_reward",
        feature_components=("observation_one_hot", "prev_reward"),
    ),
    NeuralContextMode.PREV_REWARD_TIME: _NeuralContextModeSpec(
        display_label="obs+prev_reward+time",
        feature_components=(
            "observation_one_hot",
            "normalized_time_spent",
            "prev_reward",
        ),
    ),
    NeuralContextMode.PREV_ACTION_REWARD_TIME: _NeuralContextModeSpec(
        display_label="obs+prev_action+prev_reward+time",
        feature_components=(
            "observation_one_hot",
            "normalized_time_spent",
            "prev_reward",
            "prev_action_one_hot",
        ),
    ),
}
DEFAULT_NEURAL_CONTEXT_MODE: Final[NeuralContextMode] = (
    NeuralContextMode.PREV_ACTION_REWARD_TIME
)
NEURAL_CONTEXT_MODES: Final[tuple[NeuralContextMode, ...]] = tuple(
    NeuralContextMode
)


def validate_context_mode(context_mode: str | NeuralContextMode) -> NeuralContextMode:
    try:
        return NeuralContextMode(context_mode)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported context_mode {context_mode!r}. Expected one of "
            f"{', '.join(NEURAL_CONTEXT_MODES)}."
        ) from exc


def context_mode_artifact_label(context_mode: str | NeuralContextMode) -> str:
    return validate_context_mode(context_mode).value


def context_mode_display_label(context_mode: str | NeuralContextMode) -> str:
    return _NEURAL_CONTEXT_MODE_SPECS[
        validate_context_mode(context_mode)
    ].display_label


def context_mode_feature_components(
    context_mode: str | NeuralContextMode,
) -> tuple[str, ...]:
    return _NEURAL_CONTEXT_MODE_SPECS[
        validate_context_mode(context_mode)
    ].feature_components


def context_mode_feature_dim(
    context_mode: str | NeuralContextMode,
    *,
    observation_dim: int,
    action_dim: int,
) -> int:
    """Return the neural input width produced by a context mode.

    observation_dim is the length of the state one-hot vector, and action_dim
    is the length of the previous-action one-hot vector. Time spent and reward
    each add one scalar feature when included by the selected mode.
    """
    component_widths = {
        "observation_one_hot": int(observation_dim),
        "normalized_time_spent": 1,
        "prev_reward": 1,
        "prev_action_one_hot": int(action_dim),
    }
    return sum(
        component_widths[component]
        for component in context_mode_feature_components(context_mode)
    )
