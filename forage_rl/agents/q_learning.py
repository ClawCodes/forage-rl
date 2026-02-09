"""Q-learning agents for reinforcement learning."""

from typing import List, Optional

import numpy as np
from forage_rl import Trajectory

from .base import BaseAgent

from forage_rl.config import DefaultParams
from forage_rl.environments import Maze
from forage_rl import TimedTransition
from ..types import Transition

LOG_PROB_EPSILON = 1e-12


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
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize tabular Q-learning with epsilon-greedy exploration."""
        super().__init__(maze, seed=seed, rng=rng)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = np.zeros((maze.num_states, maze.num_actions))
        # TODO: should the following be tracked for all Q-agents?
        self.q_history: List[List[float]] = [
            [] for _ in range(maze.num_states * maze.num_actions)
        ]
        self.returns: List[float] = []

    def choose_action(self, state_idx: int) -> int:
        """Choose action using epsilon-greedy exploration."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.maze.num_actions))
        return int(np.argmax(self.q_table[state_idx]))

    def update_q_value(self, transition: Transition):
        """Update Q-value using TD learning."""
        best_next_action_idx = np.argmax(self.q_table[transition.next_state])
        td_target = transition.reward + self.gamma * self.q_table[
            transition.next_state, best_next_action_idx
        ]
        td_error = td_target - self.q_table[transition.state, transition.action]
        self.q_table[transition.state, transition.action] += self.alpha * td_error

    def train(self, verbose: bool = True):
        """Train the agent."""
        for episode in range(self.num_episodes):
            state_idx = self.maze.reset()
            done = False
            total_reward = 0

            while not done:
                action_idx = self.choose_action(state_idx)
                transition, done = self.maze.step(action_idx)
                self.update_q_value(transition)
                state_idx = transition.next_state
                total_reward += transition.reward

            # Track Q-values for analysis
            for s in range(self.maze.num_states):
                for a in range(self.maze.num_actions):
                    self.q_history[s * self.maze.num_actions + a].append(
                        self.q_table[s, a]
                    )

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.returns.append(total_reward)

        if verbose:
            print("Training completed.")
            print("Final Q-table:")
            print(self.q_table)


class QLearningTime(BaseAgent):
    """
    Q-learning agent with time-aware state representation. (i.e. Q(s, t, a))

    Uses Boltzmann exploration and incorporates time spent in state.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        alpha: float = DefaultParams.ALPHA,
        gamma: float = DefaultParams.GAMMA,
        beta: float = DefaultParams.BETA,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize time-indexed Q-learning with Boltzmann exploration."""
        super().__init__(maze, beta, seed=seed, rng=rng)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))

    def choose_action(self, state_idx: int, time_spent_idx: int) -> int:
        """Choose action using Boltzmann exploration."""
        return self.choose_action_boltzmann(self.q_table[state_idx, time_spent_idx])

    def update_q_value(self, timed_transition: TimedTransition):
        """Update Q-value using TD learning."""
        if timed_transition.state == timed_transition.next_state:
            next_time_spent_idx = min(
                timed_transition.time_spent + 1, self.maze.horizon - 1
            )
        else:
            next_time_spent_idx = 0

        best_next_action_idx = np.argmax(
            self.q_table[timed_transition.next_state, next_time_spent_idx]
        )
        td_target = (
            timed_transition.reward
            + self.gamma
            * self.q_table[
                timed_transition.next_state,
                next_time_spent_idx,
                best_next_action_idx,
            ]
        )
        td_error = (
            td_target
            - self.q_table[
                timed_transition.state,
                timed_transition.time_spent,
                timed_transition.action,
            ]
        )
        self.q_table[
            timed_transition.state,
            timed_transition.time_spent,
            timed_transition.action,
        ] += self.alpha * td_error

    def simulate_q_learning(
        self, timed_trajectory: Trajectory[TimedTransition]
    ) -> list[float]:
        """Evaluate log-likelihood of transitions under Q-learning updates.

        Args:
            timed_trajectory: Trajectory containing only TimedTransition entries.

        Returns:
            List of log-likelihoods for each transition
        """
        log_likelihoods = []

        for transition in timed_trajectory.transitions:
            if not isinstance(transition, TimedTransition):
                raise TypeError(
                    "TimedTransition required; plain Transition trajectories are unsupported for time-indexed simulation."
                )
            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(
                self.q_table[transition.state, transition.time_spent]
            )
            selected_prob = float(
                np.clip(action_probs[transition.action], LOG_PROB_EPSILON, 1.0)
            )
            log_likelihoods.append(np.log(selected_prob))

            # Update Q-table
            self.update_q_value(transition)

        return log_likelihoods

    def train(self, verbose: bool = True) -> Trajectory:
        """Train the agent and optionally save trajectories.

        Args:
            verbose: Whether to print progress

        Returns:
            List of transitions
        """
        transitions = []

        for episode in range(self.num_episodes):
            state_idx = self.maze.reset()
            time_spent_idx = 0
            done = False
            max_time_spent = 0

            while not done:
                action_idx = self.choose_action(state_idx, time_spent_idx)
                transition, done = self.maze.step(action_idx)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent_idx
                )

                transitions.append(timed_transition)

                self.update_q_value(timed_transition)

                next_state_idx = timed_transition.next_state
                if state_idx == next_state_idx:
                    time_spent_idx += 1
                else:
                    time_spent_idx = 0

                state_idx = next_state_idx
                max_time_spent = max(max_time_spent, time_spent_idx)

            if verbose:
                print(f"Episode {episode}, max time spent: {max_time_spent}")
                if episode % 100 == 0:
                    avg_q = np.mean(self.q_table)
                    print(f"Episode {episode}, Average Q-value: {avg_q:.4f}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)
