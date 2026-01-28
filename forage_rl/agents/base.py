"""Base agent class with shared functionality."""

import matplotlib.pyplot as plt
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
        self.q_table = None  # Subclasses must initialize

    def boltzmann_action_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Compute Boltzmann (softmax) action probabilities."""
        exp_values = np.exp(q_values * self.beta)
        return exp_values / np.sum(exp_values)

    def choose_action_boltzmann(self, q_values: np.ndarray) -> int:
        """Choose action using Boltzmann exploration."""
        action_probs = self.boltzmann_action_probs(q_values)
        return np.random.choice(len(q_values), p=action_probs)

    def plot_q_values(self, max_time_to_display: int = 6, show: bool = True):
        """Plot Q-values as heatmaps for each action.

        Args:
            max_time_to_display: Maximum time steps to show
            show: Whether to call plt.show()
        """
        if self.q_table is None or len(self.q_table.shape) != 3:
            raise ValueError("Q-table must be 3D: [states, time, actions]")

        max_time = min(
            max_time_to_display, self.q_table.shape[1]
        )  # 2nd dimension is time
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        actions = ["Stay", "Leave"]

        vmin = np.min(self.q_table[:, :max_time, :])
        vmax = np.max(self.q_table[:, :max_time, :])

        for i, action in enumerate([0, 1]):
            im = axes[i].imshow(
                self.q_table[:, :max_time, action], vmin=vmin, vmax=vmax
            )
            axes[i].set_title(f"Q-values for Action {action} ({actions[i]})")
            axes[i].set_xlabel("Time Spent")
            axes[i].set_ylabel("States")
            axes[i].set_xticks(range(max_time))
            axes[i].set_xticklabels([f"t={j}" for j in range(max_time)])
            axes[i].set_yticks(range(self.maze.num_states))

            state_labels = (
                ["Upper Patch", "Lower Patch"]
                if self.maze.num_states == 2
                else [f"State {s}" for s in range(self.maze.num_states)]
            )
            axes[i].set_yticklabels(state_labels)

            # Annotate cells
            for y in range(self.maze.num_states):
                for x in range(max_time):
                    value = self.q_table[y, x, action]
                    color = (
                        "w"
                        if (vmax - vmin) > 0 and (value - vmin) / (vmax - vmin) > 0.5
                        else "black"
                    )
                    axes[i].text(
                        x,
                        y,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

            fig.colorbar(im, ax=axes[i])

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        if self.q_table is None:
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
