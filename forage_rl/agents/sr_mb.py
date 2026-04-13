"""SR-MB: Successor Representation computed from a learned one-step transition model."""

import numpy as np

from forage_rl import TimedTransition, Trajectory
from forage_rl.agents.successor_base import BaseSRAgent
from forage_rl.config import DefaultParams
from forage_rl.environments import Maze


class SRMBAgent(BaseSRAgent):
    """
    SR-MB (Algorithm 2 in Russek et al. 2017).

    Learns an explicit one-step transition model T_pi(s,s') = Σ_a π(a|s)·P(s'|s,a)
    and recomputes M analytically at each decision point:

        M = (I − γ·T_pi)^{-1}     [Eq 12]

    Because a local update to T_pi (one row) is immediately reflected in the
    global matrix solve, SR-MB can adapt to transition changes (detour tasks)
    without replanning from every state. However, M is computed relative to the
    cached behavioral policy π, so novel policy revaluation tasks (reward placed
    in a region π never visits) will fail.
    """

    def __init__(
        self,
        maze: Maze,
        num_episodes: int = DefaultParams.NUM_EPISODES,
        gamma: float = DefaultParams.GAMMA,
        alpha_sr: float = DefaultParams.ALPHA_SR,
        alpha_w: float = DefaultParams.ALPHA_W,
        alpha_pi: float = DefaultParams.ALPHA_PI,
        beta: float = DefaultParams.BETA,
    ):
        super().__init__(maze, num_episodes, gamma, alpha_sr, alpha_w, beta)
        self.alpha_pi = alpha_pi

        n = maze.observation_space.n  # type: ignore
        n_a = maze.num_actions

        # π[s, a_global]: policy distribution, initialized uniform over valid actions
        self.pi = np.zeros((n, n_a))
        for s in range(n):
            valid = self.q_table.valid_actions(s)
            self.pi[s, valid] = 1.0 / len(valid)

        # One-step policy-weighted transition matrix T_pi(s, s')
        self.T_pi = np.zeros((n, n))
        self._rebuild_T_pi()
        self._recompute_M()

    def _rebuild_T_pi(self) -> None:
        """Recompute T_pi(s,s') = Σ_a π(a|s)·P(s'|s,a) for all states."""
        n = self.maze.observation_space.n  # type: ignore
        T = np.zeros((n, n))
        for s in range(n):
            for a in self.q_table.valid_actions(s):
                for ns, prob in self._get_transitions(s, a):
                    T[s, ns] += self.pi[s, a] * prob
        self.T_pi = T

    def _recompute_M(self) -> None:
        """M = (I − γ·T_pi)^{-1} via linear solve for numerical stability (Eq 12)."""
        n = self.maze.observation_space.n  # type: ignore
        self.M = np.linalg.solve(np.eye(n) - self.gamma * self.T_pi, np.eye(n))

    def update_pi(self, state: int, action: int) -> None:
        """Delta-rule update of π(·|state) toward observed action, then renormalize."""
        n_a = self.maze.num_actions
        e_a = np.zeros(n_a)
        e_a[action] = 1.0
        self.pi[state] = (1 - self.alpha_pi) * self.pi[state] + self.alpha_pi * e_a
        # Renormalize over valid actions only
        valid = self.q_table.valid_actions(state)
        total = self.pi[state, valid].sum()
        if total > 0:
            self.pi[state, valid] /= total

    def _step_update(self, t: TimedTransition) -> None:
        """Apply all updates for a single observed transition."""
        self.update_r(t.state, t.action, t.reward, t.time_spent)
        self.update_pi(t.state, t.action)
        self._rebuild_T_pi()
        self._recompute_M()
        self.update_w(t.state, t.reward)
        self.update_q_cache()

    def train(self, verbose: bool = True) -> Trajectory:
        """Train the agent and return the collected trajectory."""
        transitions = []

        for episode in range(self.num_episodes):
            state, _ = self.maze.reset()
            time_spent = 0
            done = False

            while not done:
                local_idx = self.choose_action_boltzmann(
                    self.q_table.action_values(state, time_spent)
                )
                action = self.q_table.local_to_global(state, local_idx)
                transition, done = self.maze.step_transition(action)

                timed_t = TimedTransition.from_transition_time(transition, time_spent)
                transitions.append(timed_t)
                self._step_update(timed_t)

                time_spent = time_spent + 1 if state == timed_t.next_state else 0
                state = timed_t.next_state

            if verbose and episode % 50 == 0:
                print(f"Episode {episode}")

        if verbose:
            print("Training completed.")
            self.print_policy()

        return Trajectory(transitions=transitions)

    def simulate(self, trajectory: Trajectory) -> list[float]:
        """Return per-transition log-likelihoods and update internal state."""
        log_likelihoods = []

        for t in trajectory.transitions:
            action_probs = self.boltzmann_action_probs(
                self.q_table.action_values(t.state, t.time_spent)
            )
            local_idx = self.q_table.global_to_local(t.state, t.action)
            log_likelihoods.append(np.log(action_probs[local_idx]))
            self._step_update(t)

        return log_likelihoods
