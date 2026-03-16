"""Model-based reinforcement learning agent."""

import numpy as np

from forage_rl import ObservedTimedTransition, TimedTransition, Trajectory
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze, MazePOMDP

from .base import BaseAgent


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
        seed: int | None = None,
    ):
        super().__init__(maze=maze, beta=beta, seed=seed)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.num_planning_steps = num_planning_steps
        self.q_table = np.zeros((maze.agent_num_states, maze.horizon, maze.num_actions))
        self.r_table = np.zeros((maze.agent_num_states, maze.horizon, maze.num_actions))
        self.count = np.zeros((maze.agent_num_states, maze.horizon, maze.num_actions))
        self.transition_counts = np.zeros(
            (maze.agent_num_states, maze.num_actions, maze.agent_num_states)
        )
        self._use_learned_transitions = isinstance(maze, MazePOMDP)

    def _transition_outcomes(
        self,
        state: int,
        time_spent: int,
        action: int,
    ) -> list[tuple[int, int, float]]:
        """Return (next_state, next_time_spent, prob) outcomes for planning."""
        if self._use_learned_transitions:
            counts = self.transition_counts[state, action]
            total = counts.sum()
            if total == 0:
                fallback_outcomes = []
                for fallback_next_state in range(self.maze.agent_num_states):
                    try:
                        fallback_next_time = self.maze.next_time_spent(
                            state=state,
                            next_state=fallback_next_state,
                            time_spent=time_spent,
                            action=action,
                        )
                    except ValueError:
                        continue
                    fallback_outcomes.append((fallback_next_state, fallback_next_time))

                if not fallback_outcomes:
                    raise ValueError(
                        "No structurally valid observation transitions for "
                        f"state={state}, action={action}"
                    )
                fallback_prob = 1.0 / len(fallback_outcomes)
                return [
                    (next_state, next_time, fallback_prob)
                    for next_state, next_time in fallback_outcomes
                ]

            outcomes = []
            for next_state, count in enumerate(counts):
                if count == 0:
                    continue
                next_time = self.maze.next_time_spent(
                    state=state,
                    next_state=next_state,
                    time_spent=time_spent,
                    action=action,
                )
                outcomes.append((next_state, next_time, float(count / total)))
            return outcomes

        outcomes = []
        for next_state, prob in self.maze.transition_distribution(state, action):
            next_time = self.maze.next_time_spent(
                state=state,
                next_state=next_state,
                time_spent=time_spent,
                action=action,
            )
            outcomes.append((next_state, next_time, prob))
        return outcomes

    def _transition_terminates_episode(
        self,
        state: int,
        time_spent: int,
        action: int,
        next_state: int,
    ) -> bool:
        """Return whether a planned transition reaches the episode horizon."""
        timing = self.maze._transition_timing(
            state=state,
            next_state=next_state,
            action=action,
        )
        return time_spent + timing.elapsed_time >= self.maze.horizon

    def q_value_iteration(self):
        """Perform Q-value iteration using learned rewards and known transitions."""
        for _ in range(self.num_planning_steps):
            for s in range(self.maze.agent_num_states):
                for t in range(self.maze.horizon):
                    for a in range(self.maze.num_actions):
                        r_sa = self.r_table[s, t, a]
                        continuation_value = sum(
                            0.0
                            if self._transition_terminates_episode(s, t, a, next_state)
                            else prob * np.max(self.q_table[next_state, next_time])
                            for next_state, next_time, prob in self._transition_outcomes(
                                s, t, a
                            )
                        )
                        self.q_table[s, t, a] = r_sa + self.gamma * continuation_value

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """
        Evaluate replay log-likelihoods under model-based RL.

        Args:
            trajectory: Terminal-aware timed trajectory data.

        Returns:
            List of log-likelihoods for each transition
        """
        self.validate_replay_trajectory(trajectory)
        log_likelihoods = []

        for transition in trajectory.transitions:
            state = transition.state
            action = transition.action
            reward = transition.reward
            next_state = transition.next_state
            time_spent = transition.time_spent
            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(self.q_table[state, time_spent])
            log_likelihoods.append(np.log(action_probs[action]))

            # Update reward estimate with running average
            self.count[state, time_spent, action] += 1
            self.r_table[state, time_spent, action] += (
                reward - self.r_table[state, time_spent, action]
            ) / self.count[state, time_spent, action]
            if self._use_learned_transitions:
                self.transition_counts[state, action, next_state] += 1

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
                action = self.choose_action_boltzmann(self.q_table[state, time_spent])
                details = self.maze.step_transition_details(action)
                if self.maze.agent_num_states == self.maze.true_num_states:
                    timed_transition = TimedTransition.from_transition_time(
                        details.transition, time_spent, done=details.done
                    )
                else:
                    timed_transition = (
                        ObservedTimedTransition.from_transition_time_truth(
                            details.transition,
                            time_spent,
                            true_state=details.true_state,
                            true_next_state=details.true_next_state,
                            done=details.done,
                        )
                    )

                transitions.append(timed_transition)

                # Update reward estimate
                self.count[state, time_spent, action] += 1
                self.r_table[state, time_spent, action] += (
                    timed_transition.reward - self.r_table[state, time_spent, action]
                ) / self.count[state, time_spent, action]
                if self._use_learned_transitions:
                    self.transition_counts[
                        state, action, timed_transition.next_state
                    ] += 1

                next_state = timed_transition.next_state
                time_spent = self.maze.next_time_spent(
                    state=state,
                    next_state=next_state,
                    time_spent=time_spent,
                    action=action,
                    true_state=details.true_state,
                    true_next_state=details.true_next_state,
                )

                state = next_state
                done = details.done

                # Perform planning after each transition
                self.q_value_iteration()

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)
