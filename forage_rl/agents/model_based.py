"""Model-based reinforcement learning agent."""

import numpy as np
from .base import BaseAgent
from .q_table import QTable

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
        self.q_table = QTable(maze, timed=True)
        self.r_table = QTable(maze, timed=True)
        self.count = QTable(maze, timed=True)

    def q_value_iteration(self):
        """Perform Q-value iteration using learned rewards and known transitions."""
        get_transitions = (
            self.maze.obs_transition_distribution
            if self.maze.observable
            else self.maze.transition_distribution
        )
        for _ in range(self.num_planning_steps):
            for s in range(self.maze.observation_space.n):  # type: ignore
                for t in range(self.maze.horizon):
                    for a in self.q_table.valid_actions(s):
                        r_sa = self.r_table.get(s, a, t)
                        next_q = sum(
                            prob
                            * self.q_table.max_value(
                                ns,
                                min(t + 1, self.maze.horizon - 1) if ns == s else 0,
                            )
                            for ns, prob in get_transitions(s, a)
                        )
                        self.q_table.set(s, a, r_sa + self.gamma * next_q, t)

    def simulate(self, trajectory: Trajectory) -> list[float]:
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
            ai = self.q_table.global_to_local(state, action)
            action_probs = self.boltzmann_action_probs(
                self.q_table.action_values(state, time_spent)
            )
            log_likelihoods.append(np.log(action_probs[ai]))

            # Update reward estimate with running average
            self.count.update(state, action, 1.0, time_spent)
            n = self.count.get(state, action, time_spent)
            delta = (reward - self.r_table.get(state, action, time_spent)) / n
            self.r_table.update(state, action, delta, time_spent)

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

            state, _ = self.maze.reset()
            time_spent = 0
            done = False

            while not done:
                # Choose action using Boltzmann exploration
                local_idx = self.choose_action_boltzmann(
                    self.q_table.action_values(state, time_spent)
                )
                action = self.q_table.local_to_global(state, local_idx)
                transition, done = self.maze.step_transition(action)

                timed_transition = TimedTransition.from_transition_time(
                    transition, time_spent
                )

                transitions.append(timed_transition)

                # Update reward estimate
                self.count.update(state, action, 1.0, time_spent)
                n = self.count.get(state, action, time_spent)
                delta = (
                    timed_transition.reward
                    - self.r_table.get(state, action, time_spent)
                ) / n
                self.r_table.update(state, action, delta, time_spent)

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
