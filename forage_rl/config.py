"""Centralized configuration for directory paths and hyperparameters."""

from math import isfinite
from pathlib import Path
from typing import ClassVar, Final

__all__ = (
    "BASE_DIR",
    "PACKAGE_DIR",
    "DATA_DIR",
    "TRAJECTORIES_DIR",
    "LOGPROBS_DIR",
    "CHECKPOINTS_DIR",
    "FIGURES_DIR",
    "MAZE_SPECS_DIR",
    "OUTPUT_DIRS",
    "ensure_output_directories",
    "DefaultParams",
)

######################
# Needed Directories #
######################

PACKAGE_DIR: Final[Path] = (
    Path(__file__).resolve().parent
)  # forage_rl package directory
BASE_DIR: Final[Path] = PACKAGE_DIR.parent  # Repository root directory

DATA_DIR: Final[Path] = BASE_DIR / "data"  # Root directory for generated data
TRAJECTORIES_DIR: Final[Path] = DATA_DIR / "trajectories"  # Saved trajectory files
LOGPROBS_DIR: Final[Path] = DATA_DIR / "logprobs"  # Saved log-probability arrays
CHECKPOINTS_DIR: Final[Path] = DATA_DIR / "checkpoints"  # Saved model checkpoints
FIGURES_DIR: Final[Path] = BASE_DIR / "outputs" / "figures"  # Generated figures

MAZE_SPECS_DIR: Final[Path] = (
    PACKAGE_DIR / "environments" / "maze_specs"
)  # Built-in maze spec TOML files

OUTPUT_DIRS: Final[tuple[Path, ...]] = (
    TRAJECTORIES_DIR,
    LOGPROBS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
)  # Generated directories to create before writing artifacts


def ensure_output_directories() -> None:
    """Create generated data and figure directories if they don't exist."""
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


######################
#  Hyperparameters   #
######################


class DefaultParams:
    """Default constants used across training, inference, and planning."""

    # Learning parameters
    ALPHA: ClassVar[float] = 0.1  # Tabular Q-learning rate
    GAMMA: ClassVar[float] = 0.9  # Discount factor
    EPSILON: ClassVar[float] = 0.1  # Exploration rate (epsilon-greedy)
    BETA: ClassVar[float] = 1.0  # Inverse temperature for Boltzmann exploration

    # Training parameters
    TRAINING_EPISODES: ClassVar[int] = 200  # Episodes for one agent.train() call
    NUM_RUN_DATASETS: ClassVar[int] = 100  # Independent run datasets to generate

    # Environment parameters
    HORIZON: ClassVar[int] = 1000  # Maximum timesteps per episode

    # Value iteration parameters
    NUM_PLANNING_STEPS: ClassVar[int] = 10  # Planning updates per model-based step
    CONVERGENCE_THRESHOLD: ClassVar[float] = 0.01  # Value iteration stop tolerance

    # Successor representation parameters
    ALPHA_SR: ClassVar[float] = 0.3  # SR matrix learning rate
    ALPHA_W: ClassVar[float] = 0.3  # Reward weight learning rate
    ALPHA_PI: ClassVar[float] = 0.1  # Policy learning rate (SR-MB only)
    K_REPLAY: ClassVar[int] = 10  # Replay steps per transition (SR-Dyna)

    # Neural agent parameters
    NEURAL_INPUT_SCHEMA_VERSION: ClassVar[int] = 2  # Neural checkpoint input version
    NEURAL_LEARNING_RATE: ClassVar[float] = 1e-3  # Adam learning rate
    REPLAY_CAPACITY: ClassVar[int] = 4096  # Maximum replay buffer transitions
    BATCH_SIZE: ClassVar[int] = 64  # Replay minibatch size
    RECURRENT_HIDDEN_SIZE: ClassVar[int] = 64  # Recurrent hidden-state width
    RECURRENT_NUM_LAYERS: ClassVar[int] = 1  # Recurrent network layer count
    RECURRENT_SEQUENCE_LENGTH: ClassVar[int] = 16  # Replay sequence length
    RECURRENT_BURN_IN: ClassVar[int] = 0  # Warm-up steps before recurrent loss
    TARGET_UPDATE_INTERVAL: ClassVar[int] = 50  # Steps between target-network syncs
    GRADIENT_CLIP: ClassVar[float] = 5.0  # Maximum gradient norm
    DEFAULT_SEED: ClassVar[int] = 0  # Default deterministic seed

    @classmethod
    def validate(cls) -> None:
        """Validate default hyperparameters and raise ValueError on invalid values."""
        for name in (
            "TRAINING_EPISODES",
            "NUM_RUN_DATASETS",
            "HORIZON",
            "NUM_PLANNING_STEPS",
            "K_REPLAY",
            "NEURAL_INPUT_SCHEMA_VERSION",
            "REPLAY_CAPACITY",
            "BATCH_SIZE",
            "RECURRENT_HIDDEN_SIZE",
            "RECURRENT_NUM_LAYERS",
            "RECURRENT_SEQUENCE_LENGTH",
            "TARGET_UPDATE_INTERVAL",
        ):
            cls._require_positive_int(name, getattr(cls, name))

        cls._require_non_negative_int("RECURRENT_BURN_IN", cls.RECURRENT_BURN_IN)
        cls._require_non_negative_int("DEFAULT_SEED", cls.DEFAULT_SEED)

        for name in (
            "ALPHA",
            "BETA",
            "CONVERGENCE_THRESHOLD",
            "ALPHA_SR",
            "ALPHA_W",
            "ALPHA_PI",
            "NEURAL_LEARNING_RATE",
            "GRADIENT_CLIP",
        ):
            cls._require_positive_float(name, getattr(cls, name))

        cls._require_float_range("EPSILON", cls.EPSILON, minimum=0.0, maximum=1.0)
        cls._require_float_range(
            "GAMMA",
            cls.GAMMA,
            minimum=0.0,
            maximum=1.0,
            include_minimum=False,
        )

        if cls.BATCH_SIZE > cls.REPLAY_CAPACITY:
            raise ValueError(
                "BATCH_SIZE must be less than or equal to REPLAY_CAPACITY."
            )
        if cls.RECURRENT_BURN_IN >= cls.RECURRENT_SEQUENCE_LENGTH:
            raise ValueError(
                "RECURRENT_BURN_IN must be less than RECURRENT_SEQUENCE_LENGTH."
            )

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _require_non_negative_int(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")

    @staticmethod
    def _require_positive_float(name: str, value: float) -> None:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not isfinite(float(value))
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be a finite positive float, got {value!r}.")

    @staticmethod
    def _require_float_range(
        name: str,
        value: float,
        *,
        minimum: float,
        maximum: float,
        include_minimum: bool = True,
        include_maximum: bool = True,
    ) -> None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite float, got {value!r}.")

        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{name} must be a finite float, got {value!r}.")

        minimum_ok = (
            numeric_value >= minimum if include_minimum else numeric_value > minimum
        )
        maximum_ok = (
            numeric_value <= maximum if include_maximum else numeric_value < maximum
        )
        if not minimum_ok or not maximum_ok:
            lower_bracket = "[" if include_minimum else "("
            upper_bracket = "]" if include_maximum else ")"
            raise ValueError(
                f"{name} must be in range "
                f"{lower_bracket}{minimum}, {maximum}{upper_bracket}, got {value!r}."
            )


DefaultParams.validate()
