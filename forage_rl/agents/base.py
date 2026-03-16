"""Base agent class with shared functionality."""

from abc import ABC, abstractmethod

import numpy as np

from forage_rl import TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class BaseAgent(ABC):
    """
    Base class for reinforcement learning agents.

    Provides shared functionality for Boltzmann action selection and Q-value plotting.
    """

    def __init__(
        self,
        maze: Maze,
        beta: float = DefaultParams.BETA,
        seed: int | None = None,
    ):
        """
        Initialize common agent state.

        Args:
            maze: Environment instance the agent interacts with.
            beta: Inverse temperature for Boltzmann exploration.
            seed: Optional seed to create a agent-local RNG.
                If ``None``, behavior is not guaranteed to be
                reproducible across runs.
        """
        self.maze = maze
        self.beta = beta
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_table = np.array([])  # Subclasses must initialize

    def boltzmann_action_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Compute Boltzmann (softmax) action probabilities."""
        scaled_values = np.asarray(q_values, dtype=float) * self.beta
        shifted_values = scaled_values - np.max(scaled_values)
        exp_values = np.exp(shifted_values)
        return exp_values / np.sum(exp_values)

    def choose_action_boltzmann(self, q_values: np.ndarray) -> int:
        """Choose action using Boltzmann exploration."""
        action_probs = self.boltzmann_action_probs(q_values)
        return int(self.rng.choice(len(q_values), p=action_probs))

    def validate_replay_trajectory(self, trajectory: Trajectory) -> None:
        """Reject replay inputs that do not carry timed terminal-aware transitions."""
        if not trajectory.terminal_flags_present:
            raise ValueError(
                "Replay requires terminal-aware TimedTransition trajectories. "
                "Regenerate trajectories with terminal flags before running replay."
            )
        if any(
            not isinstance(transition, TimedTransition) for transition in trajectory
        ):
            raise TypeError("Replay requires TimedTransition trajectory data")

    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        if self.q_table.size == 0:
            raise ValueError("Q-table not initialized")

        if self.q_table.ndim == 3:
            return np.argmax(self.q_table, axis=2)  # time-bound q-table
        elif self.q_table.ndim == 2:
            return np.argmax(self.q_table, axis=1)  # non-time-bound q-table
        else:
            raise ValueError(
                "Q-table must have shape (num_states, num_actions) or (num_states, horizon, num_actions)"
            )

    @abstractmethod
    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate replay log-likelihoods for terminal-aware timed transitions."""
        ...

    @abstractmethod
    def train(self, verbose: bool = True) -> Trajectory:
        """Train the agent on the provided Maze."""
        ...

    def print_policy(self, max_time_to_display: int = 6):
        """Print the optimal policy for each state."""
        policy = self.get_policy()
        action_labels = getattr(self.maze, "action_labels", None)

        def action_label_for(action_idx: int) -> str:
            if hasattr(self.maze, "get_action_label"):
                return str(self.maze.get_action_label(action_idx))
            if action_labels is not None:
                return str(action_labels[action_idx])
            return str(action_idx)

        if len(policy.shape) == 2:
            max_time = min(max_time_to_display, policy.shape[1])
            print("Optimal Policy:")
            for s in range(policy.shape[0]):
                print(f"State {s}:")
                for t in range(max_time):
                    action = action_label_for(int(policy[s, t]))
                    print(f"  Time {t}: {action}")
        else:
            print("Optimal Policy:")
            for s in range(policy.shape[0]):
                action = action_label_for(int(policy[s]))
                print(f"State {s}: {action}")
