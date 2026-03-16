"""Time-aware Q-learning agent used by the experiment pipeline."""

import numpy as np

from forage_rl import ObservedTimedTransition, TimedTransition, Trajectory
from forage_rl.config import DefaultParams

from .base import BaseAgent


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
        super().__init__(maze=maze, beta=beta, seed=seed)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((maze.agent_num_states, maze.horizon, maze.num_actions))

    def choose_action(self, state: int, time_spent: int) -> int:
        """Choose action using Boltzmann exploration."""
        return self.choose_action_boltzmann(self.q_table[state, time_spent])

    def update_q_value(self, t: TimedTransition):
        """Update Q-value using TD learning."""
        next_time_spent = self.maze.next_time_spent(
            state=t.state,
            next_state=t.next_state,
            time_spent=t.time_spent,
            action=t.action,
            true_state=getattr(t, "true_state", None),
            true_next_state=getattr(t, "true_next_state", None),
        )
        if t.done:
            td_target = t.reward
        else:
            best_next_action = np.argmax(self.q_table[t.next_state, next_time_spent])
            td_target = (
                t.reward
                + self.gamma
                * self.q_table[t.next_state, next_time_spent, best_next_action]
            )
        td_error = td_target - self.q_table[t.state, t.time_spent, t.action]
        self.q_table[t.state, t.time_spent, t.action] += self.alpha * td_error

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Evaluate replay log-likelihoods under Q-learning updates.

        Args:
            trajectory: Terminal-aware timed trajectory data.

        Returns:
            List of log-likelihoods for each transition
        """
        self.validate_replay_trajectory(trajectory)
        log_likelihoods = []

        for t in trajectory.transitions:
            # Compute log-likelihood under current policy
            action_probs = self.boltzmann_action_probs(
                self.q_table[t.state, t.time_spent]
            )
            log_likelihoods.append(np.log(action_probs[t.action]))

            # Update Q-table
            self.update_q_value(t)

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
            state, _ = self.maze.reset()
            time_spent = 0
            done = False
            max_time_spent = 0

            while not done:
                action = self.choose_action(state, time_spent)
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

                self.update_q_value(timed_transition)

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
                max_time_spent = max(max_time_spent, time_spent)
                done = details.done

            if verbose:
                print(f"Episode {episode}, max time spent: {max_time_spent}")
                if episode % 100 == 0:
                    avg_q = np.mean(self.q_table)
                    print(f"Episode {episode}, Average Q-value: {avg_q:.4f}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)


__all__ = ["QLearningTime"]
