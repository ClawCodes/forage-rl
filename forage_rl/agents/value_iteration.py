"""Exact value iteration solver and agent wrapper for fully observable mazes."""

from __future__ import annotations

import numpy as np

from forage_rl import TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze, MazePOMDP

from .base import BaseAgent


class ValueIterationSolver:
    """Finite-horizon dynamic-programming solver for fully observable mazes."""

    def __init__(
        self,
        maze: Maze,
        gamma: float = DefaultParams.GAMMA,
        convergence_threshold: float = DefaultParams.CONVERGENCE_THRESHOLD,
    ):
        if isinstance(maze, MazePOMDP):
            raise ValueError("value_iteration only supports fully observable mazes")
        self.maze = maze
        self.gamma = gamma
        self.threshold = convergence_threshold
        self.V = np.zeros((maze.agent_num_states, maze.horizon), dtype=float)
        self.q_table = np.zeros(
            (maze.agent_num_states, maze.horizon, maze.num_actions),
            dtype=float,
        )
        self.policy = np.zeros((maze.agent_num_states, maze.horizon), dtype=int)

    def _transition_reward(
        self,
        *,
        state: int,
        next_state: int,
        time_spent: int,
    ) -> float:
        if state != next_state:
            return 0.0
        return float(np.exp(-self.maze.decays[state] * time_spent))

    def _transition_terminates(
        self,
        *,
        state: int,
        next_state: int,
        time_spent: int,
        action: int,
    ) -> bool:
        elapsed_time = self.maze.transition_duration(state, action, next_state)
        return time_spent + elapsed_time >= self.maze.horizon

    def _compute_action_value(self, state: int, time_spent: int, action: int) -> float:
        expected_value = 0.0
        for next_state, prob in self.maze.transition_distribution(state, action):
            reward = self._transition_reward(
                state=state,
                next_state=next_state,
                time_spent=time_spent,
            )
            continuation = 0.0
            if not self._transition_terminates(
                state=state,
                next_state=next_state,
                time_spent=time_spent,
                action=action,
            ):
                next_time_spent = self.maze.next_time_spent(
                    state=state,
                    next_state=next_state,
                    time_spent=time_spent,
                    action=action,
                )
                continuation = self.gamma * self.V[next_state, next_time_spent]
            expected_value += prob * (reward + continuation)
        return expected_value

    def solve(self, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Solve the finite-horizon control problem by backward induction."""
        del verbose  # Retained for API compatibility with the previous solver.

        for time_spent in range(self.maze.horizon - 1, -1, -1):
            for state in range(self.maze.agent_num_states):
                for action in range(self.maze.num_actions):
                    self.q_table[state, time_spent, action] = (
                        self._compute_action_value(
                            state,
                            time_spent,
                            action,
                        )
                    )
                self.V[state, time_spent] = float(
                    np.max(self.q_table[state, time_spent])
                )
                self.policy[state, time_spent] = int(
                    np.argmax(self.q_table[state, time_spent])
                )

        return self.V, self.policy

    def print_value_function(self) -> None:
        """Print the value function for each state and time step."""
        print("Value Function:")
        for state in range(self.maze.agent_num_states):
            print(f"State {state}:")
            for time_spent in range(self.maze.horizon):
                print(f"  Time {time_spent}: {self.V[state, time_spent]:.2f}")

    def print_policy(self) -> None:
        """Print the optimal policy for each state and time step."""
        print("\nOptimal Policy:")
        for state in range(self.maze.agent_num_states):
            print(f"State {state} ({self.maze.get_state_label(state)}):")
            for time_spent in range(self.maze.horizon):
                action = self.maze.get_action_label(int(self.policy[state, time_spent]))
                print(f"  Time {time_spent}: {action}")


class ValueIterationAgent(BaseAgent):
    """Exact-planning baseline exposed through the shared agent interface."""

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        beta: float = DefaultParams.BETA,
        convergence_threshold: float = DefaultParams.CONVERGENCE_THRESHOLD,
        seed: int | None = None,
    ):
        if isinstance(maze, MazePOMDP):
            raise ValueError("value_iteration only supports fully observable mazes")
        super().__init__(maze=maze, beta=beta, seed=seed)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold
        self.solver = ValueIterationSolver(
            maze=maze,
            gamma=gamma,
            convergence_threshold=convergence_threshold,
        )
        self.solver.solve(verbose=False)
        self.q_table = self.solver.q_table.copy()

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate replay log-likelihoods under the fixed exact policy."""
        self.validate_replay_trajectory(trajectory)
        log_likelihoods: list[float] = []
        for transition in trajectory.transitions:
            action_probs = self.boltzmann_action_probs(
                self.q_table[transition.state, transition.time_spent]
            )
            log_likelihoods.append(float(np.log(action_probs[transition.action])))
        return log_likelihoods

    def train(self, verbose: bool = True) -> Trajectory:
        """Roll out the fixed exact policy and return the resulting trajectory."""
        transitions: list[TimedTransition] = []
        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            done = False

            while not done:
                action = self.choose_action_boltzmann(self.q_table[state, time_spent])
                details = self.maze.step_transition_details(action)
                timed_transition = TimedTransition.from_transition_time(
                    details.transition,
                    time_spent,
                    done=details.done,
                )
                transitions.append(timed_transition)

                state = timed_transition.next_state
                time_spent = self.maze.next_time_spent(
                    state=timed_transition.state,
                    next_state=timed_transition.next_state,
                    time_spent=time_spent,
                    action=timed_transition.action,
                )
                done = timed_transition.done

            if verbose:
                print(f"Episode {episode}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)


__all__ = ["ValueIterationAgent", "ValueIterationSolver"]
