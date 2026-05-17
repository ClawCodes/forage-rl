"""Centralized configuration for paths and hyperparameters."""

from pathlib import Path
from typing import ClassVar, Final

# Base directories
BASE_DIR: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
TRAJECTORIES_DIR: Final[Path] = DATA_DIR / "trajectories"
LOGPROBS_DIR: Final[Path] = DATA_DIR / "logprobs"
CHECKPOINTS_DIR: Final[Path] = DATA_DIR / "checkpoints"
FIGURES_DIR: Final[Path] = BASE_DIR / "outputs" / "figures"
MAZE_SPECS_DIR: Final[Path] = BASE_DIR / "forage_rl" / "environments" / "maze_specs"


def ensure_directories() -> None:
    """Create output directories if they don't exist."""
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROBS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class DefaultParams:
    """Default constants used across training, inference, and planning."""

    # Learning parameters
    ALPHA: ClassVar[float] = 0.1  # Learning rate for Q-learning
    GAMMA: ClassVar[float] = 0.9  # Discount factor
    EPSILON: ClassVar[float] = 0.1  # Exploration rate (epsilon-greedy)
    BETA: ClassVar[float] = 1.0  # Inverse temperature for Boltzmann exploration

    # Training parameters
    NUM_EPISODES: ClassVar[int] = 200
    NUM_TRAINING_RUNS: ClassVar[int] = 100  # Number of trajectory files to generate
    NUM_TRAINING_EPISODES: ClassVar[int] = 1  # Episodes per generated run dataset

    # Environment parameters
    HORIZON: ClassVar[int] = 100  # Maximum timesteps per episode
    MAX_TIME_SPENT: ClassVar[int] = 10

    # Value iteration parameters
    NUM_PLANNING_STEPS: ClassVar[int] = 10
    CONVERGENCE_THRESHOLD: ClassVar[float] = 0.01

    # Successor representation parameters
    ALPHA_SR: ClassVar[float] = 0.3  # SR matrix learning rate
    ALPHA_W: ClassVar[float] = 0.3  # Reward weight learning rate
    ALPHA_PI: ClassVar[float] = 0.1  # Policy learning rate (SR-MB only)
    K_REPLAY: ClassVar[int] = 10  # Replay steps per transition (SR-Dyna)

    # Neural agent parameters
    NEURAL_FEATURE_SCHEMA_VERSION: ClassVar[int] = 2
    LEARNING_RATE: ClassVar[float] = 1e-3
    REPLAY_CAPACITY: ClassVar[int] = 4096
    BATCH_SIZE: ClassVar[int] = 64
    RECURRENT_HIDDEN_SIZE: ClassVar[int] = 64
    RECURRENT_NUM_LAYERS: ClassVar[int] = 1
    RECURRENT_SEQUENCE_LENGTH: ClassVar[int] = 16
    RECURRENT_BURN_IN: ClassVar[int] = 0
    TARGET_UPDATE_INTERVAL: ClassVar[int] = 50
    GRADIENT_CLIP: ClassVar[float] = 5.0
    FRESH_EVALUATOR_SEED: ClassVar[int] = 0
