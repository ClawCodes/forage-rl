"""Centralized configuration for paths and hyperparameters."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"
LOGPROBS_DIR = DATA_DIR / "logprobs"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MAZE_SPECS_DIR = BASE_DIR / "forage_rl" / "environments" / "maze_specs"


def ensure_directories():
    """Create output directories if they don't exist."""
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROBS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Default hyperparameters
class DefaultParams:
    """Default constants used across training, inference, and planning."""

    # Learning parameters
    ALPHA = 0.1  # Learning rate for Q-learning
    GAMMA = 0.9  # Discount factor
    EPSILON = 0.1  # Exploration rate (epsilon-greedy)
    BETA = 1.0  # Inverse temperature for Boltzmann exploration

    # Training parameters
    NUM_EPISODES = 200
    NUM_TRAINING_RUNS = 100  # Number of trajectory files to generate
    NUM_TRAINING_EPISODES = 6  # Episodes per training run for trajectory generation

    # Environment parameters
    HORIZON = 100  # Maximum timesteps per episode

    # Value iteration parameters
    NUM_PLANNING_STEPS = 10
    CONVERGENCE_THRESHOLD = 0.01
    MAX_TIME_SPENT = 10
