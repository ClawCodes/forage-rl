"""Value iteration solver for the foraging maze."""

from typing import Optional

import numpy as np

from forage_rl.config import DefaultParams, MazeParams


class ValueIterationSolver:
    """Computes optimal value function and policy via dynamic programming.

    Assumes full knowledge of transition dynamics and reward function.
    """

    def __init__(
        self,
        num_states: int = 6,
        max_time_spent: Optional[int] = None,
        gamma: Optional[float] = None,
        decays: Optional[list] = None,
        convergence_threshold: Optional[float] = None,
    ):
        self.num_states = num_states
        self.max_time_spent = max_time_spent or DefaultParams.MAX_TIME_SPENT
        self.gamma = gamma or DefaultParams.GAMMA
        self.decays = decays or (
            MazeParams.FULL_MAZE_DECAYS
            if num_states == 6
            else MazeParams.SIMPLE_MAZE_DECAYS
        )
        self.threshold = convergence_threshold or DefaultParams.CONVERGENCE_THRESHOLD

        self.upper_patch = list(range(num_states // 2))
        self.lower_patch = list(range(num_states // 2, num_states))

        self.V = np.zeros((num_states, self.max_time_spent))
        self.policy = np.zeros((num_states, self.max_time_spent), dtype=int)

    def _compute_stay_value(self, state: int, time: int) -> float:
        """Compute value of staying in current state."""
        r_sa = np.exp(-self.decays[state] * time)
        if time + 1 < self.max_time_spent:
            return r_sa + self.gamma * self.V[state, time + 1]
        return r_sa

    def _compute_leave_value(self, state: int) -> float:
        """Compute value of leaving current patch."""
        r_sa = 0.0  # No reward during travel
        probs = MazeParams.TRANSITION_PROBS

        if state in self.upper_patch:
            # Transition to lower patch
            targets = self.lower_patch
        else:
            # Transition to upper patch
            targets = self.upper_patch

        if self.num_states == 2:
            # Deterministic transition for SimpleMaze
            target = 1 - state
            return r_sa + self.gamma * self.V[target, 0]
        else:
            # Stochastic transitions for full maze
            expected_value = (
                probs[0] * self.V[targets[0], 0]
                + probs[1] * self.V[targets[1], 0]
                + probs[2] * self.V[targets[2], 0]
            )
            return r_sa + self.gamma * expected_value

    def solve(self, verbose: bool = True) -> tuple:
        """Run value iteration until convergence.

        Returns:
            Tuple of (value_function, policy)
        """
        delta = float("inf")
        iterations = 0

        while delta > self.threshold:
            delta = 0
            for s in range(self.num_states):
                for t in range(self.max_time_spent):
                    val_stay = self._compute_stay_value(s, t)
                    val_leave = self._compute_leave_value(s)

                    v_new = max(val_stay, val_leave)
                    delta = max(delta, abs(v_new - self.V[s, t]))
                    self.V[s, t] = v_new

            iterations += 1

        # Extract policy
        for s in range(self.num_states):
            for t in range(self.max_time_spent):
                val_stay = self._compute_stay_value(s, t)
                val_leave = self._compute_leave_value(s)
                self.policy[s, t] = 0 if val_stay >= val_leave else 1

        if verbose:
            print(f"Converged in {iterations} iterations")

        return self.V, self.policy

    def print_value_function(self):
        """Print the value function for each state and time step."""
        print("Value Function:")
        for s in range(self.num_states):
            print(f"State {s}:")
            for t in range(self.max_time_spent):
                print(f"  Time {t}: {self.V[s, t]:.2f}")

    def print_policy(self):
        """Print the optimal policy for each state and time step."""
        print("\nOptimal Policy:")
        for s in range(self.num_states):
            print(f"State {s}:")
            for t in range(self.max_time_spent):
                action = "Stay" if self.policy[s, t] == 0 else "Leave"
                print(f"  Time {t}: {action}")

    def get_initial_value(self, state: int = 0) -> float:
        """Get expected value of starting in given state at time 0."""
        return self.V[state, 0]


if __name__ == "__main__":
    # Solve for the full 6-state maze
    print("=" * 50)
    print("Solving for 6-state maze")
    print("=" * 50)
    solver = ValueIterationSolver(num_states=6)
    solver.solve()
    solver.print_value_function()
    solver.print_policy()
    print(f"\nExpected value from state 0: {solver.get_initial_value(0):.2f}")

    # Solve for the simple 2-state maze
    print("\n" + "=" * 50)
    print("Solving for 2-state maze")
    print("=" * 50)
    solver_simple = ValueIterationSolver(
        num_states=2, decays=MazeParams.SIMPLE_MAZE_DECAYS
    )
    solver_simple.solve()
    solver_simple.print_value_function()
    solver_simple.print_policy()
    print(f"\nExpected value from state 0: {solver_simple.get_initial_value(0):.2f}")
