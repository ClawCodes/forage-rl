"""Model-based reinforcement learning agent."""

from typing import Optional

import numpy as np
from .base import BaseAgent

from forage_rl.config import DefaultParams
from forage_rl import TimedTransition, Trajectory
from forage_rl.environments import Maze

LOG_PROB_EPSILON = 1e-12


class MBRL(BaseAgent):
    """
    Model-based RL using value iteration with known dynamics and learned rewards.

    The agent learns the reward function through experience while assuming
    the transition dynamics are known. After each observation, it performs
    value iteration to update its Q-values.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        num_planning_steps: int = DefaultParams.NUM_PLANNING_STEPS,
        beta: float = DefaultParams.BETA,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize model-based agent with learned rewards and planning buffers."""
        super().__init__(maze, beta, seed=seed, rng=rng)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.num_planning_steps = num_planning_steps
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))
        self.r_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))
        self.visit_counts = np.zeros((maze.num_states, maze.horizon, maze.num_actions))

    def q_value_iteration(self):
        """Perform Q-value iteration using learned rewards and known transitions."""
        for _ in range(self.num_planning_steps):
            for state_idx in range(self.maze.num_states):
                for time_spent_idx in range(self.maze.horizon):
                    for action_idx in range(self.maze.num_actions):
                        estimated_reward = self.r_table[
                            state_idx, time_spent_idx, action_idx
                        ]

                        expected_next_value = 0.0
                        for next_state_idx, transition_prob in self.maze.transition_distribution(
                            state_idx, action_idx
                        ):
                            transition_duration = self.maze.transition_duration(
                                state_idx, action_idx, next_state_idx
                            )
                            if next_state_idx == state_idx:
                                next_time_spent_idx = min(
                                    time_spent_idx + transition_duration,
                                    self.maze.horizon - 1,
                                )
                            else:
                                next_time_spent_idx = 0

                            expected_next_value += transition_prob * np.max(
                                self.q_table[next_state_idx, next_time_spent_idx]
                            )

                        self.q_table[state_idx, time_spent_idx, action_idx] = (
                            estimated_reward + self.gamma * expected_next_value
                        )

    def simulate_model_based_rl(
        self, timed_trajectory: Trajectory[TimedTransition]
    ) -> list[float]:
        """
        Evaluate log-likelihood of transitions under model-based RL.

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
            state_idx = transition.state
            action_idx = transition.action
            reward_value = transition.reward
            time_spent_idx = transition.time_spent

            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(
                self.q_table[state_idx, time_spent_idx]
            )
            selected_prob = float(
                np.clip(action_probs[action_idx], LOG_PROB_EPSILON, 1.0)
            )
            log_likelihoods.append(np.log(selected_prob))

            # Update reward estimate with running average
            self.visit_counts[state_idx, time_spent_idx, action_idx] += 1
            self.r_table[state_idx, time_spent_idx, action_idx] += (
                reward_value - self.r_table[state_idx, time_spent_idx, action_idx]
            ) / self.visit_counts[state_idx, time_spent_idx, action_idx]

            # Perform planning
            self.q_value_iteration()

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
            if verbose:
                print(f"Episode {episode}")

            state_idx = self.maze.reset()
            time_spent_idx = 0
            done = False

            while not done:
                # Choose action using Boltzmann exploration
                action_idx = self.choose_action_boltzmann(
                    self.q_table[state_idx, time_spent_idx]
                )
                transition, done = self.maze.step(action_idx)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent_idx
                )

                transitions.append(timed_transition)

                # Update reward estimate
                self.visit_counts[state_idx, time_spent_idx, action_idx] += 1
                self.r_table[state_idx, time_spent_idx, action_idx] += (
                    timed_transition.reward
                    - self.r_table[state_idx, time_spent_idx, action_idx]
                ) / self.visit_counts[state_idx, time_spent_idx, action_idx]

                next_state_idx = timed_transition.next_state
                if state_idx == next_state_idx:
                    time_spent_idx += 1
                else:
                    time_spent_idx = 0

                state_idx = next_state_idx

                # Perform planning after each transition
                self.q_value_iteration()

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)
