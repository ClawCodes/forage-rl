"""Base agent class with shared functionality."""

from typing import Optional

import numpy as np

from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class BaseAgent:
    """Base class for reinforcement learning agents.

    Provides shared functionality for Boltzmann action selection and Q-value plotting.
    """

    def __init__(
        self,
        maze: Maze,
        beta: float = DefaultParams.BETA,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize shared agent state and stochastic policy sampler."""
        self.maze = maze
        self.beta = beta
        self.q_table = np.array([])  # Subclasses must initialize
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def boltzmann_action_probs(self, action_values: np.ndarray) -> np.ndarray:
        """Compute Boltzmann (softmax) action probabilities."""
        values = np.asarray(action_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("action_values must be a non-empty 1D array")

        scaled_values = values * self.beta
        if not np.all(np.isfinite(scaled_values)):
            raise ValueError("action_values must be finite for Boltzmann sampling")

        shifted_values = scaled_values - np.max(scaled_values)
        exp_values = np.exp(shifted_values)
        total = float(np.sum(exp_values))
        if not np.isfinite(total) or total <= 0:
            return np.full(values.shape, 1.0 / values.size, dtype=float)

        action_probs = exp_values / total
        if not np.all(np.isfinite(action_probs)):
            return np.full(values.shape, 1.0 / values.size, dtype=float)

        action_probs = np.clip(action_probs, 0.0, 1.0)
        prob_sum = float(np.sum(action_probs))
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            return np.full(values.shape, 1.0 / values.size, dtype=float)
        return action_probs / prob_sum

    def choose_action_boltzmann(self, action_values: np.ndarray) -> int:
        """Choose action using Boltzmann exploration."""
        action_probs = self.boltzmann_action_probs(action_values)
        return int(self.rng.choice(len(action_values), p=action_probs))

    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        if self.q_table.size == 0:
            raise ValueError("Q-table not initialized")

        if len(self.q_table.shape) == 3:
            return np.argmax(self.q_table, axis=2)  # time-bound q-table
        else:
            return np.argmax(self.q_table, axis=1)  # non-time-bound q-table

    def print_policy(self, max_time_to_display: int = 6):
        """Print the optimal policy for each state."""
        policy = self.get_policy()

        if len(policy.shape) == 2:
            max_time = min(max_time_to_display, policy.shape[1])
            print("Optimal Policy:")
            for s in range(self.maze.num_states):
                print(f"State {s}:")
                for t in range(max_time):
                    action = "Stay" if policy[s, t] == 0 else "Leave"
                    print(f"  Time {t}: {action}")
        else:
            print("Optimal Policy:")
            for s in range(self.maze.num_states):
                action = "Stay" if policy[s] == 0 else "Leave"
                print(f"State {s}: {action}")
