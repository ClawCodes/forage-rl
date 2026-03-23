"""Q-learning agents for reinforcement learning."""

import numpy as np

from forage_rl import RunDataset, TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze

from ..types import Transition
from .base import BaseAgent
from .q_table import QTable


class QLearning(BaseAgent):
    """Basic Q-learning agent without time dimension. (i.e. Q(s, a))

    Uses epsilon-greedy exploration.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        alpha: float = DefaultParams.ALPHA,
        gamma: float = DefaultParams.GAMMA,
        epsilon: float = DefaultParams.EPSILON,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
        seed: int | None = None,
    ):
        super().__init__(maze, seed=seed)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = QTable(maze, timed=False)
        self.q_history: list[list[float]] = [
            [] for _ in range(self.q_table.n_obs * self.maze.num_actions)
        ]
        self.returns: list[float] = []

    def simulate(self, trajectory) -> list[float]:
        raise NotImplementedError(
            "QLearning does not support simulation; use QLearningTime."
        )

    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy exploration."""
        if self.rng.random() < self.epsilon:
            local_idx = int(self.rng.choice(self.q_table.num_valid_actions(state)))
            return self.q_table.local_to_global(state, local_idx)
        return self.q_table.best_global_action(state)

    def update_q_value(self, t: Transition):
        """Update Q-value using TD learning."""
        td_target = t.reward + self.gamma * self.q_table.max_value(t.next_state)
        td_error = td_target - self.q_table.get(t.state, t.action)
        self.q_table.update(t.state, t.action, self.alpha * td_error)

    def train(self, verbose: bool = True) -> RunDataset:
        """Train the agent and return one trajectory per episode."""
        trajectories: list[Trajectory[Transition]] = []

        for _ in range(self.num_episodes):
            state, _ = self.maze.reset()
            done = False
            total_reward = 0.0
            episode_transitions: list[Transition] = []

            while not done:
                action = self.choose_action(state)
                transition, done = self.maze.step_transition(action)
                episode_transitions.append(transition)
                self.update_q_value(transition)
                state = transition.next_state
                total_reward += transition.reward

            dense_q = self.q_table.to_array()
            for s in range(dense_q.shape[0]):
                for a in range(dense_q.shape[1]):
                    self.q_history[s * self.maze.num_actions + a].append(
                        float(dense_q[s, a])
                    )

            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.returns.append(total_reward)
            trajectories.append(Trajectory(transitions=episode_transitions))

        if verbose:
            print("Training completed.")
            print("Final Q-table:")
            print(self.q_table.to_array())

        return RunDataset(trajectories=trajectories)


class QLearningTime(BaseAgent):
    """
    Q-learning agent with time-aware state representation. (i.e. Q(s, t, a))

    Uses Boltzmann exploration and incorporates time spent in state.
    """

    def __init__(
        self,
        maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        alpha: float = DefaultParams.ALPHA,
        gamma: float = DefaultParams.GAMMA,
        beta: float = DefaultParams.BETA,
        seed: int | None = None,
    ):
        super().__init__(maze, beta, seed=seed)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = QTable(maze, timed=True)

    def choose_action(self, state: int, time_spent: int) -> int:
        """Choose action using Boltzmann exploration. Returns local action index."""
        return self.choose_action_boltzmann(
            self.q_table.action_values(state, time_spent)
        )

    def update_q_value(self, t: TimedTransition):
        """Update Q-value using TD learning."""
        if t.state == t.next_state:
            next_time_spent = min(t.time_spent + 1, self.maze.horizon - 1)
        else:
            next_time_spent = 0

        td_target = t.reward + self.gamma * self.q_table.max_value(
            t.next_state, next_time_spent
        )
        td_error = td_target - self.q_table.get(t.state, t.action, t.time_spent)
        self.q_table.update(t.state, t.action, self.alpha * td_error, t.time_spent)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate log-likelihood of transitions under Q-learning updates.

        Args:
            trajectory: Instance of Trajectory

        Returns:
            List of log-likelihoods for each transition
        """
        log_likelihoods = []

        for t in trajectory.transitions:
            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(
                self.q_table.action_values(t.state, t.time_spent)
            )
            log_likelihoods.append(
                np.log(action_probs[self.q_table.global_to_local(t.state, t.action)])
            )

            # Update Q-table
            self.update_q_value(t)

        return log_likelihoods

    def train(self, verbose: bool = True) -> RunDataset:
        """Train the agent and optionally save trajectories.

        Args:
            verbose: Whether to print progress

        Returns:
            Run dataset containing one trajectory per episode
        """
        trajectories: list[Trajectory[TimedTransition]] = []

        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent = 0
            episode_transitions: list[TimedTransition] = []

            while not done:
                local_idx = self.choose_action(state, time_spent)
                action = self.q_table.local_to_global(state, local_idx)
                transition, done = self.maze.step_transition(action)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent
                )
                episode_transitions.append(timed_transition)

                self.update_q_value(timed_transition)

                next_state = timed_transition.next_state
                if state == next_state:
                    time_spent += 1
                else:
                    time_spent = 0

                state = next_state
                max_time_spent = max(max_time_spent, time_spent)

            if verbose:
                print(f"Episode {episode}, max time spent: {max_time_spent}")
                if episode % 100 == 0:
                    avg_q = self.q_table.mean()
                    print(f"Episode {episode}, Average Q-value: {avg_q:.4f}")

            trajectories.append(Trajectory(transitions=episode_transitions))

        if verbose:
            print("Training completed.")
            self.print_policy()

        return RunDataset(trajectories=trajectories)
