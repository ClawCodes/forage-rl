"""Centralized configuration for directory paths and hyperparameters."""

from pathlib import Path
from typing import ClassVar, Final

######################
# Needed Directories #
######################

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent  # forage_rl package directory
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
    ALPHA: ClassVar[float] = 0.1  # Learning rate for Q-learning
    GAMMA: ClassVar[float] = 0.9  # Discount factor
    EPSILON: ClassVar[float] = 0.1  # Exploration rate (epsilon-greedy)
    BETA: ClassVar[float] = 1.0  # Inverse temperature for Boltzmann exploration

    # Training parameters
    NUM_EPISODES: ClassVar[int] = 200  # Default training episodes per agent
    NUM_TRAINING_RUNS: ClassVar[int] = 100  # Number of trajectory files to generate
    NUM_TRAINING_EPISODES: ClassVar[int] = 1  # Episodes per generated run dataset

    # Environment parameters
    HORIZON: ClassVar[int] = 100  # Maximum timesteps per episode
    MAX_TIME_SPENT: ClassVar[int] = 10  # Maximum tracked time spent in a patch

    # Value iteration parameters
    NUM_PLANNING_STEPS: ClassVar[int] = 10  # Planning updates per model-based step
    CONVERGENCE_THRESHOLD: ClassVar[float] = 0.01  # Value iteration stop tolerance

    # Successor representation parameters
    ALPHA_SR: ClassVar[float] = 0.3  # SR matrix learning rate
    ALPHA_W: ClassVar[float] = 0.3  # Reward weight learning rate
    ALPHA_PI: ClassVar[float] = 0.1  # Policy learning rate (SR-MB only)
    K_REPLAY: ClassVar[int] = 10  # Replay steps per transition (SR-Dyna)

    # Neural agent parameters
    NEURAL_FEATURE_SCHEMA_VERSION: ClassVar[int] = 2  # Neural feature encoding version
    LEARNING_RATE: ClassVar[float] = 1e-3  # Optimizer learning rate
    REPLAY_CAPACITY: ClassVar[int] = 4096  # Maximum replay buffer transitions
    BATCH_SIZE: ClassVar[int] = 64  # Replay minibatch size
    RECURRENT_HIDDEN_SIZE: ClassVar[int] = 64  # Recurrent hidden-state width
    RECURRENT_NUM_LAYERS: ClassVar[int] = 1  # Recurrent network layer count
    RECURRENT_SEQUENCE_LENGTH: ClassVar[int] = 16  # Replay sequence length
    RECURRENT_BURN_IN: ClassVar[int] = 0  # Warm-up steps before recurrent loss
    TARGET_UPDATE_INTERVAL: ClassVar[int] = 50  # Steps between target-network syncs
    GRADIENT_CLIP: ClassVar[float] = 5.0  # Maximum gradient norm
    FRESH_EVALUATOR_SEED: ClassVar[int] = 0  # Default seed for fresh evaluators
