"""Base agent class with shared functionality."""

import numpy as np

from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class BaseAgent:
    """Base class for reinforcement learning agents.

    Provides shared functionality for Boltzmann action selection and Q-value plotting.
    """

    def __init__(self, maze: Maze, beta: float = DefaultParams.BETA):
        self.maze = maze
        self.beta = beta
        self.q_table = np.array([])  # Subclasses must initialize

    def boltzmann_action_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Compute Boltzmann (softmax) action probabilities."""
        exp_values = np.exp(q_values * self.beta)
        return exp_values / np.sum(exp_values)

    def choose_action_boltzmann(self, q_values: np.ndarray) -> int:
        """Choose action using Boltzmann exploration."""
        action_probs = self.boltzmann_action_probs(q_values)
        return int(np.random.choice(len(q_values), p=action_probs))

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
