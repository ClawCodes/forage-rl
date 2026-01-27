"""Q-learning agents for reinforcement learning."""

import numpy as np
import matplotlib.pyplot as plt

from base import BaseAgent
from forage_rl.config import DefaultParams


class QLearning(BaseAgent):
    """Basic Q-learning agent without time dimension.

    Uses epsilon-greedy exploration.
    """

    def __init__(
        self,
        maze,
        num_episodes: int = None,
        alpha: float = None,
        gamma: float = None,
        epsilon: float = None,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
    ):
        super().__init__(maze)
        self.num_episodes = num_episodes or DefaultParams.NUM_EPISODES
        self.alpha = alpha or DefaultParams.ALPHA
        self.gamma = gamma or DefaultParams.GAMMA
        self.epsilon = epsilon or DefaultParams.EPSILON
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = np.zeros((maze.num_states, maze.num_actions))
        self.q_history = [[] for _ in range(maze.num_states * maze.num_actions)]
        self.returns = []

    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy exploration."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.maze.num_actions)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using TD learning."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def train(self, verbose: bool = True):
        """Train the agent."""
        for episode in range(self.num_episodes):
            state = self.maze.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.maze.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            # Track Q-values for analysis
            for s in range(self.maze.num_states):
                for a in range(self.maze.num_actions):
                    self.q_history[s * self.maze.num_actions + a].append(self.q_table[s, a])

            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.returns.append(total_reward)

        if verbose:
            print("Training completed.")
            print("Final Q-table:")
            print(self.q_table)

    def plot_q_values(self, show: bool = True):
        """Plot Q-values as a heatmap."""
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(self.q_table)
        ax.set_title("Q-values")
        ax.set_xlabel("Actions")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stay", "Leave"])
        ax.set_ylabel("States")
        ax.set_yticks(range(self.maze.num_states))
        ax.set_yticklabels(["Upper Patch", "Lower Patch"] if self.maze.num_states == 2 else
                          [f"State {s}" for s in range(self.maze.num_states)])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_q_values_over_time(self, show: bool = True):
        """Plot Q-value evolution over training episodes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"S{s} {'Stay' if a == 0 else 'Leave'}"
                  for s in range(self.maze.num_states)
                  for a in range(self.maze.num_actions)]

        for i, history in enumerate(self.q_history):
            if history:
                ax.plot(history, label=labels[i])

        ax.set_title("Q-values over time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Q-value")
        ax.legend()
        if show:
            plt.show()
        return fig

    def plot_returns(self, show: bool = True):
        """Plot returns over training episodes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.returns)
        ax.set_title("Returns over episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        if show:
            plt.show()
        return fig


class QLearningTime(BaseAgent):
    """Q-learning agent with time-aware state representation.

    Uses Boltzmann exploration and incorporates time spent in state.
    """

    def __init__(
        self,
        maze,
        num_episodes: int = None,
        alpha: float = None,
        gamma: float = None,
        beta: float = None,
    ):
        super().__init__(maze, beta)
        self.num_episodes = num_episodes or DefaultParams.NUM_EPISODES
        self.alpha = alpha or DefaultParams.ALPHA
        self.gamma = gamma or DefaultParams.GAMMA
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))

    def choose_action(self, state: int, time_spent: int) -> int:
        """Choose action using Boltzmann exploration."""
        return self.choose_action_boltzmann(self.q_table[state, time_spent])

    def update_q_value(self, state: int, time_spent: int, action: int, reward: float, next_state: int):
        """Update Q-value using TD learning."""
        if state == next_state:
            next_time_spent = min(time_spent + 1, self.maze.horizon - 1)
        else:
            next_time_spent = 0

        best_next_action = np.argmax(self.q_table[next_state, next_time_spent])
        td_target = reward + self.gamma * self.q_table[next_state, next_time_spent, best_next_action]
        td_error = td_target - self.q_table[state, time_spent, action]
        self.q_table[state, time_spent, action] += self.alpha * td_error

    def simulate_q_learning(self, transitions: list) -> list:
        """Evaluate log-likelihood of transitions under Q-learning updates.

        Args:
            transitions: List of (state, time_spent, action, reward, next_state) tuples

        Returns:
            List of log-likelihoods for each transition
        """
        log_likelihoods = []

        for transition in transitions:
            state, time_spent, action, reward, next_state = [int(x) if i < 4 else x
                                                              for i, x in enumerate(transition)]
            state, time_spent, action, next_state = int(state), int(time_spent), int(action), int(next_state)

            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(self.q_table[state, time_spent])
            log_likelihoods.append(np.log(action_probs[action]))

            # Update Q-table
            self.update_q_value(state, time_spent, action, reward, next_state)

        return log_likelihoods

    def train(self, save_path: str = None, verbose: bool = True) -> list:
        """Train the agent and optionally save trajectories.

        Args:
            save_path: Path to save trajectories (if provided)
            verbose: Whether to print progress

        Returns:
            List of transitions
        """
        transitions = []

        for episode in range(self.num_episodes):
            state = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent = 0

            while not done:
                action = self.choose_action(state, time_spent)
                next_state, reward, done = self.maze.step(action)

                transition = (state, time_spent, action, reward, next_state)
                transitions.append(transition)

                self.update_q_value(state, time_spent, action, reward, next_state)

                if state == next_state:
                    time_spent += 1
                else:
                    time_spent = 0

                state = next_state
                max_time_spent = max(max_time_spent, time_spent)

            if verbose:
                print(f"Episode {episode}, max time spent: {max_time_spent}")
                if episode % 100 == 0:
                    avg_q = np.mean(self.q_table)
                    print(f"Episode {episode}, Average Q-value: {avg_q:.4f}")

        if save_path:
            import numpy as np
            print(f"Saving trajectories to {save_path}")
            np.save(save_path, transitions)

        if verbose:
            print("Training completed.")
            self.print_policy()

        return transitions
