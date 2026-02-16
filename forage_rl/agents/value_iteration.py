"""Value iteration solver for the foraging maze."""

import numpy as np

from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class ValueIterationSolver:
    """
    Value iteration solver that uses a Maze instance.

    Computes optimal value function and policy using the
    maze's structure for states, rewards, and transitions.
    """

    def __init__(
        self,
        maze: Maze,
        gamma: float = DefaultParams.GAMMA,
        convergence_threshold: float = DefaultParams.CONVERGENCE_THRESHOLD,
    ):
        self.maze = maze
        self.gamma = gamma
        self.threshold = convergence_threshold

        self.V = np.zeros((maze.num_states, maze.horizon))
        self.policy = np.zeros((maze.num_states, maze.horizon))

    def _get_expected_reward(self, state: int, time: int, action: int) -> float:
        """Get expected reward for state-time-action.

        For stay action, expected reward is the probability of reward (decay curve).
        For leave action, reward is 0 (travel cost).
        """
        if action == 0:  # Stay
            decay = self.maze.decays[state]
            return np.exp(-decay * time)
        else:  # Leave
            return 0.0

    def _get_transition_probs(self, state: int, action: int) -> list[tuple[int, float]]:
        """Get list of (next_state, probability) pairs for a state-action."""
        return self.maze.transition_distribution(state, action)

    def _compute_action_value(self, state: int, time: int, action: int) -> float:
        """
        Compute expected value of taking action a in state s at time t.

        V(s) = max_a Σ_s' P(s'|s,a) * [R(s,a,s') + γ * V(s')]
        """
        expected_value = 0.0
        for next_state, prob in self._get_transition_probs(state, action):
            reward = self._get_expected_reward(state, time, action)
            if action == 0:  # Stay - time increments
                next_time = min(time + 1, self.maze.horizon - 1)
            else:  # Leave - time resets
                next_time = 0
            expected_value += prob * (
                reward + self.gamma * self.V[next_state, next_time]
            )

        return expected_value

    def solve(self, verbose: bool = True) -> tuple:
        """Run value iteration until convergence.

        Returns:
            Tuple of (value_function, policy)
        """
        delta = float("inf")
        iterations = 0

        while delta > self.threshold:
            delta = 0
            for s in range(self.maze.num_states):
                for t in range(self.maze.horizon):
                    action_values = [
                        self._compute_action_value(s, t, a)
                        for a in range(self.maze.num_actions)
                    ]

                    v_new = max(action_values)
                    delta = max(delta, abs(v_new - self.V[s, t]))
                    self.V[s, t] = v_new

            iterations += 1

        # Extract policy
        for s in range(self.maze.num_states):
            for t in range(self.maze.horizon):
                action_values = [
                    self._compute_action_value(s, t, a)
                    for a in range(self.maze.num_actions)
                ]
                self.policy[s, t] = int(np.argmax(action_values))

        if verbose:
            print(f"Converged in {iterations} iterations")

        return self.V, self.policy

    def print_value_function(self):
        """Print the value function for each state and time step."""
        print("Value Function:")
        for s in range(self.maze.num_states):
            print(f"State {s}:")
            for t in range(self.maze.horizon):
                print(f"  Time {t}: {self.V[s, t]:.2f}")

    def print_policy(self):
        """Print the optimal policy for each state and time step."""
        print("\nOptimal Policy:")
        for s in range(self.maze.num_states):
            print(f"State {s} ({self.maze.get_state_label(s)}):")
            for t in range(self.maze.horizon):
                action = self.maze.get_action_label(int(self.policy[s, t]))
                print(f"  Time {t}: {action}")
