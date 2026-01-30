"""Model-based reinforcement learning agent."""

import numpy as np
from .base import BaseAgent

from forage_rl.config import DefaultParams
from forage_rl import TimedTransition, Trajectory
from forage_rl.environments import Maze


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
    ):
        super().__init__(maze, beta)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.num_planning_steps = num_planning_steps
        self.q_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))
        self.r_table = np.zeros((maze.num_states, maze.horizon, maze.num_actions))
        self.count = np.zeros((maze.num_states, maze.horizon, maze.num_actions))

    def q_value_iteration(self):
        """Perform Q-value iteration using learned rewards and known transitions."""
        for _ in range(self.num_planning_steps):
            for s in range(self.maze.num_states):
                for t in range(self.maze.horizon):
                    for a in range(self.maze.num_actions):
                        r_sa = self.r_table[s, t, a]

                        if a == 0:  # Stay
                            next_state = s
                            next_time = min(t + 1, self.maze.horizon - 1)
                        else:  # Leave - deterministic transition for SimpleMaze
                            # TODO: only works for simple maze - move transition dynamics to maze and make general
                            # stochastic transitions
                            if s == 0:
                                next_state = 1
                            else:
                                next_state = 0
                            next_time = 0  # reset time when leaving

                        self.q_table[s, t, a] = r_sa + self.gamma * np.max(
                            self.q_table[next_state, next_time]
                        )

    def simulate_model_based_rl(self, trajectory: Trajectory) -> list[float]:
        """
        Evaluate log-likelihood of transitions under model-based RL.

        Args:
            trajectory: instance of Trajectory

        Returns:
            List of log-likelihoods for each transition
        """
        log_likelihoods = []

        for state, action, reward, next_state, time_spent in trajectory:
            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(self.q_table[state, time_spent])
            log_likelihoods.append(np.log(action_probs[action]))

            # Update reward estimate with running average
            self.count[state, time_spent, action] += 1
            self.r_table[state, time_spent, action] += (
                reward - self.r_table[state, time_spent, action]
            ) / self.count[state, time_spent, action]

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

            state = self.maze.reset()
            time_spent = 0
            done = False

            while not done:
                # Choose action using Boltzmann exploration
                action = self.choose_action_boltzmann(self.q_table[state, time_spent])
                transition, done = self.maze.step(action)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent
                )

                transitions.append(timed_transition)

                # Update reward estimate
                self.count[state, time_spent, action] += 1
                self.r_table[state, time_spent, action] += (
                    timed_transition.reward - self.r_table[state, time_spent, action]
                ) / self.count[state, time_spent, action]

                next_state = timed_transition.next_state
                if state == next_state:
                    time_spent += 1
                else:
                    time_spent = 0

                state = next_state

                # Perform planning after each transition
                self.q_value_iteration()

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)
