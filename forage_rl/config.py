"""Centralized configuration for paths and hyperparameters."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"
LOGPROBS_DIR = DATA_DIR / "logprobs"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


def ensure_directories():
    """Create output directories if they don't exist."""
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROBS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Default hyperparameters
class DefaultParams:
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


# Maze decay rates
class MazeParams:
    # Full maze (6 states): [s0, s1, s2, s3, s4, s5]
    # Upper patch: s0, s1, s2 | Lower patch: s3, s4, s5
    FULL_MAZE_DECAYS = [0.5, 2.0, 0.1, 0.1, 2.0, 3.0]

    # Simple maze (2 states): [upper, lower]
    SIMPLE_MAZE_DECAYS = [0.2, 3.0]

    # Transition probabilities for full maze
    TRANSITION_PROBS = [0.15, 0.35, 0.50]  # Cumulative: 0.15, 0.50, 1.0

    # Action Labels
    BASE_ACTIONS = ["stay", "stay", "stay", "leave", "leave", "leave"]
    SIMPLE_ACTIONS = ["stay", "leave"]

    # State Labels
    BASE_STATES = [
        "Upper Patch",
        "Upper Patch",
        "Upper Patch",
        "Lower Patch",
        "Lower Patch",
        "Lower Patch",
    ]
    SIMPLE_STATES = ["Upper Patch", "Lower Patch"]
